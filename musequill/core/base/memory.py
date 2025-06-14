"""Base memory interfaces and classes."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel
import uuid


class MemoryType(str, Enum):
    """Types of memory in the system."""
    
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    PROCEDURAL = "procedural"


class MemoryEntry(BaseModel):
    """A single memory entry."""
    
    id: str
    memory_type: MemoryType
    content: Any
    metadata: Dict[str, Any] = {}
    created_at: datetime
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    importance: float = 0.5  # 0.0 to 1.0
    access_count: int = 0
    tags: List[str] = []


class BaseMemoryStore(ABC):
    """Base class for memory storage implementations."""
    
    def __init__(self, store_id: Optional[str] = None):
        self.store_id = store_id or str(uuid.uuid4())
        self._memory_entries: Dict[str, MemoryEntry] = {}
    
    @abstractmethod
    async def store(self, memory_type: MemoryType, content: Any, **metadata) -> str:
        """Store a memory entry and return its ID."""
        pass
    
    @abstractmethod
    async def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search for memory entries."""
        pass
    
    @abstractmethod
    async def update(self, memory_id: str, content: Any, **metadata) -> bool:
        """Update a memory entry."""
        pass
    
    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        pass
    
    @abstractmethod
    async def clear(self, memory_type: Optional[MemoryType] = None) -> int:
        """Clear memory entries."""
        pass
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        return {
            "total_entries": len(self._memory_entries),
            "by_type": {
                mem_type.value: len([
                    entry for entry in self._memory_entries.values() 
                    if entry.memory_type == mem_type
                ])
                for mem_type in MemoryType
            }
        }


class InMemoryStore(BaseMemoryStore):
    """Simple in-memory implementation for testing."""
    
    async def store(self, memory_type: MemoryType, content: Any, **metadata) -> str:
        """Store a memory entry."""
        entry_id = str(uuid.uuid4())
        entry = MemoryEntry(
            id=entry_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata,
            created_at=datetime.now()
        )
        self._memory_entries[entry_id] = entry
        return entry_id
    
    async def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry."""
        entry = self._memory_entries.get(memory_id)
        if entry:
            entry.access_count += 1
        return entry
    
    async def search(
        self, 
        query: str, 
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search memory entries."""
        results = []
        for entry in self._memory_entries.values():
            if memory_type and entry.memory_type != memory_type:
                continue
            
            # Simple string matching
            if query.lower() in str(entry.content).lower():
                results.append(entry)
            
            if len(results) >= limit:
                break
        
        return results
    
    async def update(self, memory_id: str, content: Any, **metadata) -> bool:
        """Update a memory entry."""
        if memory_id in self._memory_entries:
            entry = self._memory_entries[memory_id]
            entry.content = content
            entry.metadata.update(metadata)
            entry.updated_at = datetime.now()
            return True
        return False
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        if memory_id in self._memory_entries:
            del self._memory_entries[memory_id]
            return True
        return False
    
    async def clear(self, memory_type: Optional[MemoryType] = None) -> int:
        """Clear memory entries."""
        if memory_type is None:
            count = len(self._memory_entries)
            self._memory_entries.clear()
            return count
        else:
            to_remove = [
                entry_id for entry_id, entry in self._memory_entries.items()
                if entry.memory_type == memory_type
            ]
            for entry_id in to_remove:
                del self._memory_entries[entry_id]
            return len(to_remove)