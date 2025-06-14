"""Cost tracking and budget management for OpenAI API usage."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import structlog
from .models import TokenUsage, calculate_cost, RequestMetrics

logger = structlog.get_logger(__name__)


@dataclass
class BudgetConfig:
    """Budget configuration."""
    
    daily_budget_usd: float = 100.0
    monthly_budget_usd: float = 3000.0
    alert_threshold_pct: float = 80.0
    hard_limit_enabled: bool = True
    

@dataclass
class CostSummary:
    """Cost usage summary."""
    
    total_cost: float = 0.0
    request_count: int = 0
    token_count: int = 0
    avg_cost_per_request: float = 0.0
    avg_cost_per_token: float = 0.0
    model_breakdown: Dict[str, float] = field(default_factory=dict)
    

class CostTracker:
    """Advanced cost tracking and budget management."""
    
    def __init__(self, config: Optional[BudgetConfig] = None):
        self.config = config or BudgetConfig()
        self.daily_costs: Dict[str, float] = defaultdict(float)  # date -> cost
        self.monthly_costs: Dict[str, float] = defaultdict(float)  # month -> cost
        self.model_costs: Dict[str, float] = defaultdict(float)  # model -> cost
        self.request_history: List[RequestMetrics] = []
        self._lock = asyncio.Lock()
        self._alert_callbacks: List[callable] = []
        
    async def record_usage(
        self, 
        model: str, 
        token_usage: TokenUsage, 
        duration_ms: float,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> float:
        """Record API usage and return the cost."""
        cost = calculate_cost(model, token_usage)
        
        async with self._lock:
            # Create request metrics
            metrics = RequestMetrics(
                model=model,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                token_usage=token_usage,
                cost_usd=cost,
                success=success,
                error_message=error_message
            )
            
            # Store metrics
            self.request_history.append(metrics)
            
            # Update cost tracking
            await self._update_costs(model, cost, metrics.timestamp)
            
            # Check budget limits
            await self._check_budget_limits()
            
            logger.info(
                "Usage recorded",
                model=model,
                cost=cost,
                tokens=token_usage.total_tokens,
                success=success
            )
            
        return cost
    
    async def _update_costs(self, model: str, cost: float, timestamp: datetime) -> None:
        """Update cost tracking data."""
        date_key = timestamp.strftime("%Y-%m-%d")
        month_key = timestamp.strftime("%Y-%m")
        
        self.daily_costs[date_key] += cost
        self.monthly_costs[month_key] += cost
        self.model_costs[model] += cost
    
    async def _check_budget_limits(self) -> None:
        """Check if budget limits are exceeded and trigger alerts."""
        today = datetime.now().strftime("%Y-%m-%d")
        this_month = datetime.now().strftime("%Y-%m")
        
        daily_cost = self.daily_costs[today]
        monthly_cost = self.monthly_costs[this_month]
        
        # Check daily budget
        daily_usage_pct = (daily_cost / self.config.daily_budget_usd) * 100
        if daily_usage_pct >= self.config.alert_threshold_pct:
            await self._trigger_alert("daily", daily_usage_pct, daily_cost)
        
        # Check monthly budget
        monthly_usage_pct = (monthly_cost / self.config.monthly_budget_usd) * 100
        if monthly_usage_pct >= self.config.alert_threshold_pct:
            await self._trigger_alert("monthly", monthly_usage_pct, monthly_cost)
        
        # Hard limits
        if self.config.hard_limit_enabled:
            if daily_cost >= self.config.daily_budget_usd:
                raise BudgetExceededException(f"Daily budget exceeded: ${daily_cost:.2f}")
            if monthly_cost >= self.config.monthly_budget_usd:
                raise BudgetExceededException(f"Monthly budget exceeded: ${monthly_cost:.2f}")
    
    async def _trigger_alert(self, period: str, usage_pct: float, cost: float) -> None:
        """Trigger budget alert."""
        alert_data = {
            "period": period,
            "usage_percentage": usage_pct,
            "current_cost": cost,
            "budget": self.config.daily_budget_usd if period == "daily" else self.config.monthly_budget_usd,
            "timestamp": datetime.now()
        }
        
        logger.warning(
            "Budget alert triggered",
            **alert_data
        )
        
        # Call registered alert callbacks
        for callback in self._alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                logger.error("Alert callback failed", error=str(e))
    
    def get_daily_cost(self, date: Optional[datetime] = None) -> float:
        """Get total cost for a specific date."""
        if date is None:
            date = datetime.now()
        date_key = date.strftime("%Y-%m-%d")
        return self.daily_costs[date_key]
    
    def get_monthly_cost(self, date: Optional[datetime] = None) -> float:
        """Get total cost for a specific month."""
        if date is None:
            date = datetime.now()
        month_key = date.strftime("%Y-%m")
        return self.monthly_costs[month_key]
    
    def get_model_costs(self) -> Dict[str, float]:
        """Get cost breakdown by model."""
        return dict(self.model_costs)
    
    def get_cost_summary(self, days: int = 30) -> CostSummary:
        """Get cost summary for the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        relevant_requests = [
            req for req in self.request_history 
            if req.timestamp >= cutoff_date
        ]
        
        if not relevant_requests:
            return CostSummary()
        
        total_cost = sum(req.cost_usd for req in relevant_requests)
        request_count = len(relevant_requests)
        token_count = sum(req.token_usage.total_tokens for req in relevant_requests)
        
        model_breakdown = defaultdict(float)
        for req in relevant_requests:
            model_breakdown[req.model] += req.cost_usd
        
        return CostSummary(
            total_cost=total_cost,
            request_count=request_count,
            token_count=token_count,
            avg_cost_per_request=total_cost / request_count if request_count > 0 else 0,
            avg_cost_per_token=total_cost / token_count if token_count > 0 else 0,
            model_breakdown=dict(model_breakdown)
        )
    
    def get_usage_trend(self, days: int = 7) -> List[Tuple[str, float]]:
        """Get daily usage trend for the last N days."""
        trend = []
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            date_key = date.strftime("%Y-%m-%d")
            cost = self.daily_costs.get(date_key, 0.0)
            trend.append((date_key, cost))
        
        return list(reversed(trend))
    
    def get_budget_status(self) -> Dict[str, any]:
        """Get current budget status."""
        today = datetime.now().strftime("%Y-%m-%d")
        this_month = datetime.now().strftime("%Y-%m")
        
        daily_cost = self.daily_costs[today]
        monthly_cost = self.monthly_costs[this_month]
        
        return {
            "daily": {
                "used": daily_cost,
                "budget": self.config.daily_budget_usd,
                "remaining": max(0, self.config.daily_budget_usd - daily_cost),
                "usage_percentage": (daily_cost / self.config.daily_budget_usd) * 100,
                "alert_triggered": daily_cost >= (self.config.daily_budget_usd * self.config.alert_threshold_pct / 100)
            },
            "monthly": {
                "used": monthly_cost,
                "budget": self.config.monthly_budget_usd,
                "remaining": max(0, self.config.monthly_budget_usd - monthly_cost),
                "usage_percentage": (monthly_cost / self.config.monthly_budget_usd) * 100,
                "alert_triggered": monthly_cost >= (self.config.monthly_budget_usd * self.config.alert_threshold_pct / 100)
            }
        }
    
    def register_alert_callback(self, callback: callable) -> None:
        """Register a callback function for budget alerts."""
        self._alert_callbacks.append(callback)
    
    def clear_history(self, days: Optional[int] = None) -> None:
        """Clear request history older than N days."""
        if days is None:
            self.request_history.clear()
            self.daily_costs.clear()
            self.monthly_costs.clear()
            self.model_costs.clear()
        else:
            cutoff_date = datetime.now() - timedelta(days=days)
            self.request_history = [
                req for req in self.request_history 
                if req.timestamp >= cutoff_date
            ]
        
        logger.info("Cost history cleared", days=days)
    
    def export_metrics(self, format: str = "dict") -> any:
        """Export metrics in various formats."""
        if format == "dict":
            return {
                "config": self.config.__dict__,
                "daily_costs": dict(self.daily_costs),
                "monthly_costs": dict(self.monthly_costs),
                "model_costs": dict(self.model_costs),
                "request_count": len(self.request_history),
                "budget_status": self.get_budget_status()
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")


class BudgetExceededException(Exception):
    """Exception raised when budget limits are exceeded."""
    pass