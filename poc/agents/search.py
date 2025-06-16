import os
from threading import Lock
from typing import Any
from tavily import TavilyClient
# connect
client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

search_lock = Lock()

def search(query:str) -> Any:

    # run search
    result = client.search(query, max_results=1)

    # print first result
    data = result["results"][0]["content"]

    return result, data
