
from typing import TypedDict, List
from langchain_core.pydantic_v1 import BaseModel

class Queries(BaseModel):
    queries: List[str]

class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int