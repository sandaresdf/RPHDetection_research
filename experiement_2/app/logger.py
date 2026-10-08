import json

def save_trace(messages, metadata=None, filename="reasoning_dataset.jsonl"):

    entry = {
        "conversation": [m.content for m in messages],
        "tool_calls": [
            tc for m in messages
            if hasattr(m, "tool_calls")
            for tc in m.tool_calls or []
        ],
        "metadata": metadata
    }

    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
