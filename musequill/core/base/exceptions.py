"""Base exceptions for the MuseQuill system."""


class MuseQuillException(Exception):
    """Base exception for all MuseQuill errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class AgentException(MuseQuillException):
    """Exception raised by agents."""
    
    def __init__(self, message: str, agent_id: str = None, agent_type: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.agent_id = agent_id
        self.agent_type = agent_type


class WorkflowException(MuseQuillException):
    """Exception raised by workflows."""
    
    def __init__(self, message: str, workflow_id: str = None, step_id: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.workflow_id = workflow_id
        self.step_id = step_id


class ConfigurationException(MuseQuillException):
    """Exception raised for configuration errors."""
    pass


class APIException(MuseQuillException):
    """Exception raised for API errors."""
    
    def __init__(self, message: str, status_code: int = 500, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code


class OpenAIException(MuseQuillException):
    """Exception raised for OpenAI API errors."""
    
    def __init__(self, message: str, model: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.model = model


class MemoryException(MuseQuillException):
    """Exception raised by memory systems."""
    
    def __init__(self, message: str, memory_type: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.memory_type = memory_type


class ResearchException(MuseQuillException):
    """Exception raised by research systems."""
    
    def __init__(self, message: str, source: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.source = source


class QualityException(MuseQuillException):
    """Exception raised by quality control systems."""
    pass


class ValidationException(MuseQuillException):
    """Exception raised for validation errors."""
    
    def __init__(self, message: str, field: str = None, value: any = None, **kwargs):
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value


class AuthenticationException(MuseQuillException):
    """Exception raised for authentication errors."""
    pass


class AuthorizationException(MuseQuillException):
    """Exception raised for authorization errors."""
    pass


class RateLimitException(MuseQuillException):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class BudgetException(MuseQuillException):
    """Exception raised when budget limits are exceeded."""
    
    def __init__(self, message: str, current_cost: float = None, budget_limit: float = None, **kwargs):
        super().__init__(message, **kwargs)
        self.current_cost = current_cost
        self.budget_limit = budget_limit