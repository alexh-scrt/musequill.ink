from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import os
from enum import Enum

from poc.writer.agent import AgentState
from poc.writer.nodes import (
    plan_node,
    generation_node,
    reflection_node,
    research_plan_node,
    research_critique_node,
    should_continue
)

from poc.writer.nodes import Node

def orchestrate(memory):
    """ Orchestrate """
    builder:StateGraph = StateGraph(AgentState)
    builder.add_node(str(Node.PLANNER), plan_node)
    builder.add_node(str(Node.GENERATOR), generation_node)
    builder.add_node(str(Node.REFLECTOR), reflection_node)
    builder.add_node(str(Node.RESEARCHER), research_plan_node)
    builder.add_node(str(Node.RESEARCHER_CRITIQUE), research_critique_node)

    builder.set_entry_point(str(Node.PLANNER))

    builder.add_conditional_edges(
        str(Node.GENERATOR), 
        should_continue, 
        {END: END, str(Node.REFLECTOR): str(Node.REFLECTOR)}
    )

    builder.add_edge(str(Node.PLANNER), str(Node.RESEARCHER))
    builder.add_edge(str(Node.RESEARCHER), str(Node.GENERATOR))

    builder.add_edge(str(Node.REFLECTOR), str(Node.RESEARCHER_CRITIQUE))
    builder.add_edge(str(Node.RESEARCHER_CRITIQUE), str(Node.GENERATOR))

    graph = builder.compile(checkpointer=memory, debug=True)

    return graph