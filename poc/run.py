"""POC Research"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# pylint: disable=unused-import
import bootstrap

from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

from poc.agents.research import Agent

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

def main():
    tool = TavilySearchResults(max_results=4) #increased number of results
    print(type(tool))
    print(tool.name)

    model = ChatOpenAI(model="gpt-3.5-turbo")  #reduce inference cost
    abot = Agent(model, [tool], system=prompt)

    messages = [HumanMessage(content="What is the weather in sf?")]
    result = abot.graph.invoke({"messages": messages})
    print(result)

    print(result['messages'][-1].content)

    messages = [HumanMessage(content="What is the weather in SF and LA?")]
    result = abot.graph.invoke({"messages": messages})

    print(result['messages'][-1].content)

    # Note, the query was modified to produce more consistent results. 
    # Results may vary per run and over time as search information and models change.

    query = "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
    What is the GDP of that state? Answer each question." 
    messages = [HumanMessage(content=query)]

    model = ChatOpenAI(model="gpt-4o")  # requires more advanced model
    abot = Agent(model, [tool], system=prompt)
    result = abot.graph.invoke({"messages": messages})

    print(result['messages'][-1].content)

    print('-------------------------------------------')


if __name__ == "__main__":
    main()