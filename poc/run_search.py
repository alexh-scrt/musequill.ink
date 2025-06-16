from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import json
from pygments import highlight, lexers, formatters

# pylint: disable=unused-import
import bootstrap
from poc.agents.search import search

query = "what is the gdb of canada in 2024?"

response, content = search(query)
data = response['results'][0]
# parse JSON
parsed_json = json.loads(data)
# pretty print JSON with syntax highlighting
formatted_json = json.dumps(parsed_json, indent=4)
colorful_json = highlight(formatted_json,
                          lexers.get_lexer_for_mimetype('application/json'),
                          formatters.TerminalFormatter())

print(colorful_json)
