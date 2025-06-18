"""Main OpenAI client with advanced features."""
import os
import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from contextlib import asynccontextmanager
import tiktoken
import structlog
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types import CreateEmbeddingResponse

from .config import OpenAIConfig
from .models import TokenUsage, RequestMetrics, ModelType
from .rate_limiter import RateLimiter, RateLimitConfig
from .cost_tracker import CostTracker, BudgetConfig

logger = structlog.get_logger(__name__)


class OpenAIClient:
    """Advanced OpenAI client with rate limiting, cost tracking, and error handling."""
    
    def __init__(
        self,
        config: Optional[OpenAIConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        budget_config: Optional[BudgetConfig] = None
    ):
        self.config = config or OpenAIConfig(api_key=os.getenv('OPENAI_API_KEY'))
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout,
            max_retries=self.config.max_retries
        )
        
        self.rate_limiter = RateLimiter(rate_limit_config)
        self.cost_tracker = CostTracker(budget_config)
        self._token_encoders: Dict[str, tiktoken.Encoding] = {}
        
    def _get_encoder(self, model: str) -> tiktoken.Encoding:
        """Get or create token encoder for a model."""
        if model not in self._token_encoders:
            try:
                self._token_encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                self._token_encoders[model] = tiktoken.get_encoding("cl100k_base")
        return self._token_encoders[model]
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text for a specific model."""
        encoder = self._get_encoder(model)
        return len(encoder.encode(text))
    
    def count_message_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        """Count tokens in a list of messages."""
        encoder = self._get_encoder(model)
        
        # Base tokens per message (varies by model)
        tokens_per_message = 3 if "gpt-3.5" in model else 4
        tokens_per_name = 1
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoder.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        
        num_tokens += 3  # every reply is primed with assistant
        return num_tokens
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """Create a chat completion with full tracking."""
        model = model or self.config.default_model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        # Estimate tokens for rate limiting
        estimated_input_tokens = self.count_message_tokens(messages, model)
        estimated_total_tokens = estimated_input_tokens + max_tokens
        
        start_time = time.time()
        
        try:
            # Apply rate limiting
            async with self.rate_limiter.request_context(model, estimated_total_tokens):
                # Make the API call
                if stream:
                    return self._stream_chat_completion(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        estimated_input_tokens=estimated_input_tokens,
                        start_time=start_time,
                        **kwargs
                    )
                else:
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )
                    
                    # Record usage
                    await self._record_completion_usage(
                        model=model,
                        response=response,
                        estimated_input_tokens=estimated_input_tokens,
                        start_time=start_time
                    )
                    
                    return response
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self.cost_tracker.record_usage(
                model=model,
                token_usage=TokenUsage(prompt_tokens=estimated_input_tokens),
                duration_ms=duration_ms,
                success=False,
                error_message=str(e)
            )
            raise
    
    async def _stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        estimated_input_tokens: int,
        start_time: float,
        **kwargs
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Handle streaming chat completion."""
        completion_tokens = 0
        content_buffer = ""
        
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    content_buffer += content
                    completion_tokens += len(self._get_encoder(model).encode(content))
                
                yield chunk
            
            # Record final usage
            duration_ms = (time.time() - start_time) * 1000
            token_usage = TokenUsage(
                prompt_tokens=estimated_input_tokens,
                completion_tokens=completion_tokens,
                total_tokens=estimated_input_tokens + completion_tokens
            )
            
            await self.cost_tracker.record_usage(
                model=model,
                token_usage=token_usage,
                duration_ms=duration_ms,
                success=True
            )
            
            self.rate_limiter.update_actual_tokens(model, token_usage.total_tokens)
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self.cost_tracker.record_usage(
                model=model,
                token_usage=TokenUsage(prompt_tokens=estimated_input_tokens),
                duration_ms=duration_ms,
                success=False,
                error_message=str(e)
            )
            raise
    
    async def _record_completion_usage(
        self,
        model: str,
        response: ChatCompletion,
        estimated_input_tokens: int,
        start_time: float
    ) -> None:
        """Record usage metrics for a completed request."""
        duration_ms = (time.time() - start_time) * 1000
        
        if response.usage:
            token_usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
        else:
            # Fallback to estimation
            content = response.choices[0].message.content or ""
            completion_tokens = self.count_tokens(content, model)
            token_usage = TokenUsage(
                prompt_tokens=estimated_input_tokens,
                completion_tokens=completion_tokens,
                total_tokens=estimated_input_tokens + completion_tokens
            )
        
        await self.cost_tracker.record_usage(
            model=model,
            token_usage=token_usage,
            duration_ms=duration_ms,
            success=True
        )
        
        self.rate_limiter.update_actual_tokens(model, token_usage.total_tokens)
    
    async def create_embedding(
        self,
        input_text: Union[str, List[str]],
        model: str = ModelType.TEXT_EMBEDDING_3_SMALL,
        dimensions: Optional[int] = None,
        **kwargs
    ) -> CreateEmbeddingResponse:
        """Create embeddings with tracking."""
        start_time = time.time()
        
        if isinstance(input_text, str):
            estimated_tokens = self.count_tokens(input_text, model)
        else:
            estimated_tokens = sum(self.count_tokens(text, model) for text in input_text)
        
        try:
            async with self.rate_limiter.request_context(model, estimated_tokens):
                response = await self.client.embeddings.create(
                    model=model,
                    input=input_text,
                    dimensions=dimensions,
                    **kwargs
                )
                
                # Record usage
                duration_ms = (time.time() - start_time) * 1000
                token_usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else estimated_tokens,
                    completion_tokens=0,
                    total_tokens=response.usage.total_tokens if response.usage else estimated_tokens
                )
                
                await self.cost_tracker.record_usage(
                    model=model,
                    token_usage=token_usage,
                    duration_ms=duration_ms,
                    success=True
                )
                
                self.rate_limiter.update_actual_tokens(
                    model, 
                    token_usage.total_tokens
                )
                
                return response
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self.cost_tracker.record_usage(
                model=model,
                token_usage=TokenUsage(prompt_tokens=estimated_tokens),
                duration_ms=duration_ms,
                success=False,
                error_message=str(e)
            )
            raise
    
    async def function_call(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]],
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> ChatCompletion:
        """Make a function call with OpenAI."""
        model = model or self.config.default_model
        
        return await self.chat_completion(
            messages=messages,
            model=model,
            functions=functions,
            function_call=function_call,
            **kwargs
        )
    
    async def structured_output(
        self,
        messages: List[Dict[str, str]],
        response_format: Dict[str, Any],
        model: Optional[str] = None,
        **kwargs
    ) -> ChatCompletion:
        """Get structured output from OpenAI."""
        model = model or self.config.default_model
        
        return await self.chat_completion(
            messages=messages,
            model=model,
            response_format=response_format,
            **kwargs
        )
    
    def get_rate_limit_status(self, model: str) -> Dict[str, Any]:
        """Get current rate limit status for a model."""
        return {
            "usage": self.rate_limiter.get_current_usage(model),
            "remaining": self.rate_limiter.get_remaining_capacity(model),
            "available": self.rate_limiter.is_available(model)
        }
    
    def get_cost_status(self) -> Dict[str, Any]:
        """Get current cost and budget status."""
        return {
            "budget_status": self.cost_tracker.get_budget_status(),
            "model_costs": self.cost_tracker.get_model_costs(),
            "summary": self.cost_tracker.get_cost_summary()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the OpenAI client."""
        try:
            # Simple API call to check connectivity
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
            
            return {
                "status": "healthy",
                "api_available": True,
                "response_time_ms": getattr(response, "_response_time_ms", None),
                "model_accessible": True
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "api_available": False,
                "error": str(e),
                "model_accessible": False
            }
    
    async def close(self) -> None:
        """Close the client and cleanup resources."""
        await self.client.close()
        logger.info("OpenAI client closed")
    
    @asynccontextmanager
    async def session(self):
        """Context manager for OpenAI client sessions."""
        try:
            yield self
        finally:
            await self.close()

# Dependency injection for agent system
async def get_openai_client() -> OpenAIClient:
    """Get OpenAI client instance."""
    return OpenAIClient()

