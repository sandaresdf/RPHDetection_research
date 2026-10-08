from typing import List
import openai

class RAGAgent:

    def __init__(self, retriever):
        self.retriever = retriever

    def build_prompt(self, user_query, context_chunks):
        context = "\n".join(context_chunks)

        prompt = f"""
SYSTEM: You are a helpful university assistant.
Follow university policies.

CONTEXT:
{context}

USER:
{user_query}
"""
        return prompt

    def run(self, user_query):
        docs = self.retriever.retrieve(user_query)
        prompt = self.build_prompt(user_query, docs)

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role":"user","content":prompt}]
        )

        return response["choices"][0]["message"]["content"]
