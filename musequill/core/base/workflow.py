"""Base workflow interfaces and classes."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pydantic import BaseModel
import uuid


class WorkflowState(str, Enum):
    """Workflow execution states."""
    
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStep(BaseModel):
    """A single step in a workflow."""
    
    id: str
    name: str
    description: str
    dependencies: List[str] = []
    status: WorkflowState = WorkflowState.CREATED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


class WorkflowExecution(BaseModel):
    """Workflow execution tracking."""
    
    id: str
    workflow_id: str
    state: WorkflowState
    current_step: Optional[str] = None
    steps: Dict[str, WorkflowStep] = {}
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    context: Dict[str, Any] = {}


class BaseWorkflow(ABC):
    """Base class for workflow implementations."""
    
    def __init__(self, workflow_id: Optional[str] = None):
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.name = self.__class__.__name__
        self.description = ""
        self.steps: Dict[str, WorkflowStep] = {}
        self._step_handlers: Dict[str, Callable] = {}
    
    def add_step(
        self, 
        step_id: str, 
        name: str, 
        handler: Callable,
        description: str = "",
        dependencies: List[str] = None
    ) -> None:
        """Add a step to the workflow."""
        step = WorkflowStep(
            id=step_id,
            name=name,
            description=description,
            dependencies=dependencies or []
        )
        self.steps[step_id] = step
        self._step_handlers[step_id] = handler
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any] = None) -> WorkflowExecution:
        """Execute the workflow."""
        pass
    
    @abstractmethod
    async def pause(self, execution_id: str) -> bool:
        """Pause workflow execution."""
        pass
    
    @abstractmethod
    async def resume(self, execution_id: str) -> bool:
        """Resume workflow execution."""
        pass
    
    @abstractmethod
    async def cancel(self, execution_id: str) -> bool:
        """Cancel workflow execution."""
        pass
    
    def validate_workflow(self) -> List[str]:
        """Validate workflow structure."""
        errors = []
        
        # Check for circular dependencies
        def has_cycle(step_id: str, visited: set, rec_stack: set) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)
            
            step = self.steps.get(step_id)
            if not step:
                return False
            
            for dep in step.dependencies:
                if dep not in visited:
                    if has_cycle(dep, visited, rec_stack):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(step_id)
            return False
        
        visited = set()
        for step_id in self.steps:
            if step_id not in visited:
                if has_cycle(step_id, visited, set()):
                    errors.append(f"Circular dependency detected involving step: {step_id}")
        
        # Check for missing dependencies
        for step_id, step in self.steps.items():
            for dep in step.dependencies:
                if dep not in self.steps:
                    errors.append(f"Step {step_id} depends on non-existent step: {dep}")
        
        return errors


class SimpleWorkflow(BaseWorkflow):
    """Simple workflow implementation."""
    
    def __init__(self, workflow_id: Optional[str] = None):
        super().__init__(workflow_id)
        self._executions: Dict[str, WorkflowExecution] = {}
    
    async def execute(self, context: Dict[str, Any] = None) -> WorkflowExecution:
        """Execute the workflow."""
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            id=execution_id,
            workflow_id=self.workflow_id,
            state=WorkflowState.RUNNING,
            context=context or {},
            started_at=datetime.now()
        )
        
        self._executions[execution_id] = execution
        
        try:
            # Execute steps in dependency order
            executed_steps = set()
            
            while len(executed_steps) < len(self.steps):
                ready_steps = [
                    step_id for step_id, step in self.steps.items()
                    if step_id not in executed_steps and
                    all(dep in executed_steps for dep in step.dependencies)
                ]
                
                if not ready_steps:
                    raise RuntimeError("No ready steps found - possible circular dependency")
                
                for step_id in ready_steps:
                    await self._execute_step(execution, step_id)
                    executed_steps.add(step_id)
            
            execution.state = WorkflowState.COMPLETED
            execution.completed_at = datetime.now()
            
        except Exception as e:
            execution.state = WorkflowState.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
        
        return execution
    
    async def _execute_step(self, execution: WorkflowExecution, step_id: str) -> None:
        """Execute a single step."""
        step = self.steps[step_id]
        handler = self._step_handlers[step_id]
        
        step.status = WorkflowState.RUNNING
        step.started_at = datetime.now()
        execution.current_step = step_id
        
        try:
            result = await handler(execution.context)
            
            # Update context with step result
            if result is not None:
                execution.context[f"step_{step_id}_result"] = result
            
            step.status = WorkflowState.COMPLETED
            step.completed_at = datetime.now()
            
        except Exception as e:
            step.status = WorkflowState.FAILED
            step.error_message = str(e)
            step.completed_at = datetime.now()
            raise
    
    async def pause(self, execution_id: str) -> bool:
        """Pause workflow execution."""
        if execution_id in self._executions:
            execution = self._executions[execution_id]
            if execution.state == WorkflowState.RUNNING:
                execution.state = WorkflowState.PAUSED
                return True
        return False
    
    async def resume(self, execution_id: str) -> bool:
        """Resume workflow execution."""
        if execution_id in self._executions:
            execution = self._executions[execution_id]
            if execution.state == WorkflowState.PAUSED:
                execution.state = WorkflowState.RUNNING
                return True
        return False
    
    async def cancel(self, execution_id: str) -> bool:
        """Cancel workflow execution."""
        if execution_id in self._executions:
            execution = self._executions[execution_id]
            if execution.state in [WorkflowState.RUNNING, WorkflowState.PAUSED]:
                execution.state = WorkflowState.CANCELLED
                execution.completed_at = datetime.now()
                return True
        return False
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID."""
        return self._executions.get(execution_id)