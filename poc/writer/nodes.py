from typing import Dict, AnyStr, Any
from poc.writer.agent import AgentState, Queries
from poc.writer.prompts import (
PLAN_PROMPT, RESEARCH_PLAN_PROMPT, WRITER_PROMPT, RESEARCH_CRITIQUE_PROMPT,
REFLECTION_PROMPT,
)
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
import os
from enum import Enum

class Node(str, Enum):
    """Supported genre types."""
    # Fiction Genres
    PLANNER = "planner"
    GENERATOR = "generator"
    REFLECTOR = "reflector"
    RESEARCHER = "researcher"
    RESEARCHER_CRITIQUE = "researcher_critique"

    def __str__(self):
        return self.value

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

model = ChatOpenAI(model="gpt-4o", temperature=0)

def plan_node(state: AgentState) -> Dict[AnyStr, Any]:
    """ Planning Node"""
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}

def generation_node(state: AgentState) -> Dict[AnyStr, Any]:
    """ Generation Node """
    # Previously research critique content or empty
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
        ]
    response = model.invoke(messages)
    return {
        "draft": response.content, 
        "revision_number": state.get("revision_number", 1) + 1
    }

def research_plan_node(state: AgentState) -> Dict[AnyStr, Any]:
    """ Research Node """
    # Ask model to generate a list of queries (str), based on the prompt and task
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    # Previous content or no content 
    content = state['content'] or []
    for q in queries.queries:
        # Perform Tavily search per query
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}


def reflection_node(state: AgentState) -> Dict[AnyStr, Any]:
    """ Reflection Node """
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    # Reflect on the draft (from generation node) given the reflection prompt
    response = model.invoke(messages)
    return {"critique": response.content}

def research_critique_node(state: AgentState) -> Dict[AnyStr, Any]:
    """ Research Critique Node """
    # Research queries from critique given the prompt
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        # Search critique queries
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

def should_continue(state) -> AnyStr:
    if state["revision_number"] > state["max_revisions"]:
        return END
    return str(Node.REFLECTOR)