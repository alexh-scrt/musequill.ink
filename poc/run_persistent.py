"""POC Research"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio

# pylint: disable=unused-import
import bootstrap

from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from poc.agents.persistent import PersistentAgent

from langgraph.checkpoint.memory import MemorySaver
#from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver

#memory = AsyncSqliteSaver.from_conn_string(":memory:")

memory = MemorySaver()

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

async def main():
    tool = TavilySearchResults(max_results=4) #increased number of results
    print(type(tool))
    print(tool.name)

    model = ChatOpenAI(model="gpt-3.5-turbo")  #reduce inference cost
    abot = PersistentAgent(model, [tool], system=prompt, checkpointer=memory)

    messages = [HumanMessage(content="What is the weather in sf?")]
    thread = {"configurable": {"thread_id": "1"}}

    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(f"* Event: {v['messages']}")


    messages = [HumanMessage(content="What about in la?")]
    thread = {"configurable": {"thread_id": "1"}}
    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(f"* Event: {v['messages']}")

    messages = [HumanMessage(content="Which one is warmer?")]
    thread = {"configurable": {"thread_id": "1"}}
    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(f"* Event: {v['messages']}")


    messages = [HumanMessage(content="What is the weather in SF?")]
    thread = {"configurable": {"thread_id": "4"}}
    async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="|")

    print('-------------------------------------------')


if __name__ == "__main__":
    asyncio.run(main())