"""Rate limiting for OpenAI API requests."""

import asyncio
import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    
    requests_per_minute: int = 60
    tokens_per_minute: int = 90000
    requests_per_day: int = 10000
    tokens_per_day: int = 2000000
    burst_allowance: int = 10
    

@dataclass
class RateLimitState:
    """Current rate limit state."""
    
    requests_made: int = 0
    tokens_used: int = 0
    request_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    token_usage_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    daily_requests: int = 0
    daily_tokens: int = 0
    last_reset_date: Optional[str] = None


class RateLimiter:
    """Advanced rate limiter for OpenAI API requests."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.state_by_model: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._lock = asyncio.Lock()
        
    async def acquire(self, model: str, estimated_tokens: int = 1000) -> None:
        """Acquire permission to make a request."""
        async with self._lock:
            await self._wait_if_needed(model, estimated_tokens)
            await self._record_request(model, estimated_tokens)
    
    async def _wait_if_needed(self, model: str, estimated_tokens: int) -> None:
        """Wait if rate limits would be exceeded."""
        state = self.state_by_model[model]
        current_time = time.time()
        
        # Clean old entries
        self._clean_old_entries(state, current_time)
        
        # Check daily limits
        await self._check_daily_limits(state)
        
        # Check per-minute limits
        requests_in_minute = len(state.request_times)
        tokens_in_minute = sum(1 for _, tokens in state.token_usage_times)
        
        # Calculate wait time for request limit
        if requests_in_minute >= self.config.requests_per_minute:
            oldest_request_time = state.request_times[0]
            wait_time = 60 - (current_time - oldest_request_time)
            if wait_time > 0:
                logger.info(
                    "Rate limit reached, waiting",
                    model=model,
                    wait_time=wait_time,
                    limit_type="requests_per_minute"
                )
                await asyncio.sleep(wait_time)
        
        # Calculate wait time for token limit
        if tokens_in_minute + estimated_tokens > self.config.tokens_per_minute:
            oldest_token_time = state.token_usage_times[0][0]
            wait_time = 60 - (current_time - oldest_token_time)
            if wait_time > 0:
                logger.info(
                    "Token rate limit reached, waiting",
                    model=model,
                    wait_time=wait_time,
                    estimated_tokens=estimated_tokens,
                    limit_type="tokens_per_minute"
                )
                await asyncio.sleep(wait_time)
    
    def _clean_old_entries(self, state: RateLimitState, current_time: float) -> None:
        """Remove entries older than 1 minute."""
        cutoff_time = current_time - 60
        
        # Clean request times
        while state.request_times and state.request_times[0] < cutoff_time:
            state.request_times.popleft()
        
        # Clean token usage times
        while state.token_usage_times and state.token_usage_times[0][0] < cutoff_time:
            state.token_usage_times.popleft()
    
    async def _check_daily_limits(self, state: RateLimitState) -> None:
        """Check and reset daily limits if needed."""
        today = time.strftime("%Y-%m-%d")
        
        if state.last_reset_date != today:
            state.daily_requests = 0
            state.daily_tokens = 0
            state.last_reset_date = today
        
        if state.daily_requests >= self.config.requests_per_day:
            # Wait until tomorrow
            seconds_until_tomorrow = 86400 - (time.time() % 86400)
            logger.warning(
                "Daily request limit reached, waiting until tomorrow",
                wait_time=seconds_until_tomorrow
            )
            await asyncio.sleep(seconds_until_tomorrow)
    
    async def _record_request(self, model: str, estimated_tokens: int) -> None:
        """Record a request being made."""
        state = self.state_by_model[model]
        current_time = time.time()
        
        state.requests_made += 1
        state.daily_requests += 1
        state.request_times.append(current_time)
        state.token_usage_times.append((current_time, estimated_tokens))
        
        logger.debug(
            "Request recorded",
            model=model,
            estimated_tokens=estimated_tokens,
            total_requests=state.requests_made,
            daily_requests=state.daily_requests
        )
    
    def update_actual_tokens(self, model: str, actual_tokens: int) -> None:
        """Update with actual token usage after request completion."""
        state = self.state_by_model[model]
        state.tokens_used += actual_tokens
        state.daily_tokens += actual_tokens
        
        logger.debug(
            "Token usage updated",
            model=model,
            actual_tokens=actual_tokens,
            total_tokens=state.tokens_used,
            daily_tokens=state.daily_tokens
        )
    
    def get_current_usage(self, model: str) -> Dict[str, int]:
        """Get current usage statistics for a model."""
        state = self.state_by_model[model]
        current_time = time.time()
        
        # Clean old entries first
        self._clean_old_entries(state, current_time)
        
        return {
            "requests_per_minute": len(state.request_times),
            "tokens_per_minute": sum(tokens for _, tokens in state.token_usage_times),
            "daily_requests": state.daily_requests,
            "daily_tokens": state.daily_tokens,
            "total_requests": state.requests_made,
            "total_tokens": state.tokens_used,
        }
    
    def get_remaining_capacity(self, model: str) -> Dict[str, int]:
        """Get remaining capacity for a model."""
        usage = self.get_current_usage(model)
        
        return {
            "requests_per_minute": max(0, self.config.requests_per_minute - usage["requests_per_minute"]),
            "tokens_per_minute": max(0, self.config.tokens_per_minute - usage["tokens_per_minute"]),
            "daily_requests": max(0, self.config.requests_per_day - usage["daily_requests"]),
            "daily_tokens": max(0, self.config.tokens_per_day - usage["daily_tokens"]),
        }
    
    @asynccontextmanager
    async def request_context(self, model: str, estimated_tokens: int = 1000):
        """Context manager for making rate-limited requests."""
        await self.acquire(model, estimated_tokens)
        try:
            yield
        finally:
            pass  # Could add cleanup logic here if needed
    
    def reset_limits(self, model: Optional[str] = None) -> None:
        """Reset rate limits for a model or all models."""
        if model:
            if model in self.state_by_model:
                del self.state_by_model[model]
        else:
            self.state_by_model.clear()
        
        logger.info("Rate limits reset", model=model or "all")
    
    def is_available(self, model: str, estimated_tokens: int = 1000) -> bool:
        """Check if a request can be made immediately without waiting."""
        state = self.state_by_model[model]
        current_time = time.time()
        
        # Clean old entries
        self._clean_old_entries(state, current_time)
        
        # Check limits
        requests_in_minute = len(state.request_times)
        tokens_in_minute = sum(tokens for _, tokens in state.token_usage_times)
        
        return (
            requests_in_minute < self.config.requests_per_minute and
            tokens_in_minute + estimated_tokens <= self.config.tokens_per_minute and
            state.daily_requests < self.config.requests_per_day and
            state.daily_tokens + estimated_tokens <= self.config.tokens_per_day
        )