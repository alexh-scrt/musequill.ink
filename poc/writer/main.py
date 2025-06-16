
from pathlib import Path
import sys

from langgraph.checkpoint.memory import MemorySaver

PROJECT_ROOT_DIR = Path(__name__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT_DIR))

import bootstrap

from poc.writer.orchestrator import orchestrate

agent_start = {
        'task': "what is the difference between langchain and langsmith",
        "max_revisions": 2,
        "revision_number": 1,
        'content': [],
    }

def main():
    memory = MemorySaver()
    graph = orchestrate(memory)
    thread = {"configurable": {"thread_id": "1"}}

    print('-'*50)

    for s in graph.stream(agent_start, thread):
        print(f'EVENT: {s}\n')
        print('='*50)

    print('-'*50)

if __name__ == '__main__':
    main()