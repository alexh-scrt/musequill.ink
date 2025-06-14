"""
MuseQuill Environment Configuration
Environment-specific settings and utilities.
"""

from enum import Enum
from typing import Dict, Any, Optional
from pathlib import Path


class Environment(str, Enum):
    """Application environment types."""
    
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging" 
    PRODUCTION = "production"


class EnvironmentConfig:
    """Environment-specific configuration."""
    
    def __init__(self, env: Environment):
        self.env = env
        self.config = self._get_environment_config()
    
    def _get_environment_config(self) -> Dict[str, Any]:
        """Get configuration for the current environment."""
        configs = {
            Environment.DEVELOPMENT: {
                "debug": True,
                "log_level": "DEBUG",
                "reload": True,
                "workers": 1,
                "rate_limit_enabled": False,
                "auth_enabled": False,
                "cost_tracking_enabled": True,
                "mock_external_apis": True,
                "structured_logging": False,
                "log_format": "colored",
                "database_echo": True,
                "profiling_enabled": True,
                "cors_origins": ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"],
            },
            Environment.TESTING: {
                "debug": True,
                "log_level": "WARNING",
                "reload": False,
                "workers": 1,
                "rate_limit_enabled": False,
                "auth_enabled": True,
                "cost_tracking_enabled": False,
                "mock_external_apis": True,
                "structured_logging": True,
                "log_format": "json",
                "database_echo": False,
                "profiling_enabled": False,
                "cors_origins": [],
                "openai_api_key": "sk-test-key-for-testing-only",
            },
            Environment.STAGING: {
                "debug": False,
                "log_level": "INFO",
                "reload": False,
                "workers": 2,
                "rate_limit_enabled": True,
                "auth_enabled": True,
                "cost_tracking_enabled": True,
                "mock_external_apis": False,
                "structured_logging": True,
                "log_format": "json",
                "database_echo": False,
                "profiling_enabled": False,
                "cors_origins": ["https://staging.musequill.ink"],
            },
            Environment.PRODUCTION: {
                "debug": False,
                "log_level": "INFO",
                "reload": False,
                "workers": 4,
                "rate_limit_enabled": True,
                "auth_enabled": True,
                "cost_tracking_enabled": True,
                "mock_external_apis": False,
                "structured_logging": True,
                "log_format": "json",
                "database_echo": False,
                "profiling_enabled": False,
                "cors_origins": ["https://musequill.ink", "https://app.musequill.ink"],
                "openai_daily_budget_usd": 500.0,
                "openai_monthly_budget_usd": 15000.0,
            }
        }
        
        return configs.get(self.env, configs[Environment.DEVELOPMENT])
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get environment-specific configuration value."""
        return self.config.get(key, default)
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.env == Environment.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.env == Environment.TESTING
    
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.env == Environment.STAGING
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env == Environment.PRODUCTION
    
    def get_database_url(self, base_url: str) -> str:
        """Get environment-specific database URL."""
        if self.is_testing():
            return base_url.replace("/musequill", "/musequill_test")
        return base_url
    
    def get_redis_db(self, base_db: int = 0) -> int:
        """Get environment-specific Redis database number."""
        if self.is_testing():
            return base_db + 1
        return base_db
    
    def get_log_file_path(self, base_path: Optional[str]) -> Optional[str]:
        """Get environment-specific log file path."""
        if not base_path:
            return None
        
        path = Path(base_path)
        if self.is_testing():
            return str(path.parent / f"{path.stem}_test{path.suffix}")
        elif self.is_staging():
            return str(path.parent / f"{path.stem}_staging{path.suffix}")
        
        return base_path
    
    def get_openai_settings(self) -> Dict[str, Any]:
        """Get environment-specific OpenAI settings."""
        settings = {}
        
        if self.is_development():
            settings.update({
                "temperature": 0.8,  # More creative in development
                "max_retries": 1,  # Fewer retries for faster development
                "request_timeout": 30,  # Shorter timeout
                "daily_budget_usd": 50.0,
                "monthly_budget_usd": 1000.0,
            })
        elif self.is_testing():
            settings.update({
                "temperature": 0.0,  # Deterministic for testing
                "max_retries": 0,  # No retries in testing
                "request_timeout": 10,  # Very short timeout
                "daily_budget_usd": 10.0,
                "monthly_budget_usd": 100.0,
            })
        elif self.is_staging():
            settings.update({
                "temperature": 0.7,  # Standard creativity
                "max_retries": 2,  # Moderate retries
                "request_timeout": 45,  # Moderate timeout
                "daily_budget_usd": 200.0,
                "monthly_budget_usd": 5000.0,
            })
        elif self.is_production():
            settings.update({
                "temperature": 0.7,  # Standard creativity
                "max_retries": 3,  # Full retries for reliability
                "request_timeout": 60,  # Full timeout
                "daily_budget_usd": 500.0,
                "monthly_budget_usd": 15000.0,
            })
        
        return settings
    
    def get_security_settings(self) -> Dict[str, Any]:
        """Get environment-specific security settings."""
        settings = {}
        
        if self.is_development():
            settings.update({
                "jwt_secret_key": "dev-secret-key-not-for-production",
                "access_token_expire_minutes": 60,  # Longer for development
                "cors_allow_credentials": True,
                "cors_allow_headers": ["*"],
            })
        elif self.is_testing():
            settings.update({
                "jwt_secret_key": "test-secret-key",
                "access_token_expire_minutes": 5,  # Short for testing
                "cors_allow_credentials": False,
                "cors_allow_headers": ["Content-Type", "Authorization"],
            })
        elif self.is_production():
            settings.update({
                "access_token_expire_minutes": 30,  # Standard expiration
                "cors_allow_credentials": True,
                "cors_allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
            })
        
        return settings
    
    def get_rate_limit_settings(self) -> Dict[str, Any]:
        """Get environment-specific rate limiting settings."""
        if self.is_development() or self.is_testing():
            return {
                "enabled": False,
                "requests_per_minute": 1000,  # Very high limits
                "burst": 100,
            }
        elif self.is_staging():
            return {
                "enabled": True,
                "requests_per_minute": 120,  # Moderate limits
                "burst": 20,
            }
        elif self.is_production():
            return {
                "enabled": True,
                "requests_per_minute": 60,  # Production limits
                "burst": 10,
            }
        
        return {}
    
    def get_storage_settings(self) -> Dict[str, Any]:
        """Get environment-specific storage settings."""
        if self.is_development() or self.is_testing():
            return {
                "provider": "local",
                "local_path": f"./data/storage_{self.env.value}",
            }
        elif self.is_staging():
            return {
                "provider": "s3",
                "aws_s3_bucket": "musequill-staging-storage",
            }
        elif self.is_production():
            return {
                "provider": "s3",
                "aws_s3_bucket": "musequill-production-storage",
            }
        
        return {}
    
    def validate_environment(self) -> Dict[str, bool]:
        """Validate environment configuration."""
        validations = {
            "environment_valid": self.env in Environment,
            "config_loaded": bool(self.config),
        }
        
        # Environment-specific validations
        if self.is_production():
            validations.update({
                "debug_disabled": not self.get("debug", True),
                "auth_enabled": self.get("auth_enabled", False),
                "rate_limiting_enabled": self.get("rate_limit_enabled", False),
                "structured_logging": self.get("structured_logging", False),
            })
        
        return validations
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get environment-specific feature flags."""
        base_flags = {
            "adversarial_quality_control": True,
            "research_integration": True,
            "cost_tracking": True,
            "streaming_responses": True,
            "human_in_the_loop": True,
            "agent_metrics": True,
        }
        
        if self.is_development():
            base_flags.update({
                "debug_toolbar": True,
                "profiling": True,
                "mock_apis": True,
                "verbose_logging": True,
            })
        elif self.is_testing():
            base_flags.update({
                "mock_apis": True,
                "fast_mode": True,  # Skip expensive operations
                "deterministic_mode": True,  # Predictable outputs
            })
        elif self.is_production():
            base_flags.update({
                "performance_monitoring": True,
                "error_tracking": True,
                "health_checks": True,
            })
        
        return base_flags


def get_environment_config(env_name: str) -> EnvironmentConfig:
    """Get environment configuration for a given environment name."""
    try:
        env = Environment(env_name.lower())
        return EnvironmentConfig(env)
    except ValueError:
        # Default to development if invalid environment
        return EnvironmentConfig(Environment.DEVELOPMENT)


def detect_environment() -> Environment:
    """Detect the current environment from various sources."""
    import os
    
    # Check environment variable
    env_var = os.getenv("ENVIRONMENT", "").lower()
    if env_var:
        try:
            return Environment(env_var)
        except ValueError:
            pass
    
    # Check for environment-specific files
    current_dir = Path.cwd()
    
    if (current_dir / ".env.production").exists():
        return Environment.PRODUCTION
    elif (current_dir / ".env.staging").exists():
        return Environment.STAGING
    elif (current_dir / ".env.testing").exists():
        return Environment.TESTING
    elif (current_dir / ".env.development").exists():
        return Environment.DEVELOPMENT
    
    # Check for testing indicators
    if "pytest" in os.getenv("_", "") or "test" in os.getenv("PYTEST_CURRENT_TEST", ""):
        return Environment.TESTING
    
    # Default to development
    return Environment.DEVELOPMENT


def setup_environment_overrides(settings, env_config: EnvironmentConfig) -> None:
    """Apply environment-specific overrides to settings."""
    # Apply environment config overrides
    for key, value in env_config.config.items():
        attr_name = key.upper()
        if hasattr(settings, attr_name):
            setattr(settings, attr_name, value)
    
    # Apply OpenAI settings
    openai_settings = env_config.get_openai_settings()
    for key, value in openai_settings.items():
        attr_name = f"OPENAI_{key.upper()}"
        if hasattr(settings, attr_name):
            setattr(settings, attr_name, value)
    
    # Apply security settings
    security_settings = env_config.get_security_settings()
    for key, value in security_settings.items():
        attr_name = key.upper()
        if hasattr(settings, attr_name):
            setattr(settings, attr_name, value)
    
    # Apply rate limit settings
    rate_limit_settings = env_config.get_rate_limit_settings()
    for key, value in rate_limit_settings.items():
        attr_name = f"RATE_LIMIT_{key.upper()}"
        if hasattr(settings, attr_name):
            setattr(settings, attr_name, value)
    
    # Apply storage settings
    storage_settings = env_config.get_storage_settings()
    for key, value in storage_settings.items():
        attr_name = f"STORAGE_{key.upper()}" if not key.startswith("aws_") else key.upper()
        if hasattr(settings, attr_name):
            setattr(settings, attr_name, value)