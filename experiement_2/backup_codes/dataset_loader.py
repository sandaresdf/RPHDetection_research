from datasets import load_dataset
from langchain_core.messages import HumanMessage, SystemMessage

def load_red_queen():
    """
    Streaming avoids Arrow schema crashes.
    """
    dataset = load_dataset(
        "YifanJ/Red_Queen",
        split="train",
        streaming=True
    )
    return dataset


def get_clean_samples(dataset, n=10):
    """
    Filters out malformed rows.
    """
    samples = []

    for row in dataset:

        query = row.get("query", "")

        content_arr = []

        for message in query:
            if message.get("role") == "system":
                content_arr.append(SystemMessage(content=message.get("content", "")))
            elif message.get("role") == "user":
                content_arr.append(HumanMessage(content=message.get("content", "")))

        if not content_arr:
            continue

        samples.append(content_arr)

        if len(samples) >= n:
            break

    return samples
