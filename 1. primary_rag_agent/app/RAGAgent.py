import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Anthropic
import anthropic
from dotenv import load_dotenv

load_dotenv()


class RAGAgent:
    """
    Research-oriented RAG Agent that captures:
    - Retrieved documents
    - Externalized reasoning trace
    - Final answer

    NOTE:
    Reasoning traces are elicited explanations, NOT internal chain-of-thought.
    """

    QUERIES_LOG_DIR = Path("rag_knowledge_base/query_logs")

    def __init__(self, vector_store, model):
        self.vector_store = vector_store
        self.model = model

        # -----------------------------
        # Retrieval Tool
        # -----------------------------
        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """
            Retrieve relevant policy documents from the knowledge base.
            """
            docs = self.vector_store.similarity_search(query, k=7)

            serialized = "\n\n".join(
                f"Document {i+1} "
                f"(Category: {doc.metadata.get('category', 'unknown')}, "
                f"Source: {doc.metadata.get('source_file', 'unknown')}):\n"
                f"{doc.page_content}"
                for i, doc in enumerate(docs)
            )

            return serialized, docs

        self.retrieve_context = retrieve_context
        self.agent = self._create_agent()

    # --------------------------------------------------
    # Agent Construction
    # --------------------------------------------------
    def _create_agent(self):

        tools = [self.retrieve_context]
        system_prompt = """
You are an enterprise policy assistant for TechVenture Solutions.

FINAL_ANSWER:
- A concise, user-facing answer
"""

        agent = create_agent(
            model=self.model,
            tools=tools,
            system_prompt=system_prompt
        )

        return agent

    # --------------------------------------------------
    # Response Parsing
    # --------------------------------------------------
    def _parse_response(self, result: List) -> Dict[str, str]:
        """
        Parse structured agent output into reasoning trace and final answer.
        """
        query_result = {
            "query": None,
            "reasoning_trace": [],
            "final_answer": None,
            "retrieved_documents": [],
            "agent_output_raw": None,
            "tool_calls": None,
            "response_metadata": None,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "retrieval_top_k": 3
            }
        }

        # iterate through messages from the assistant
        for message in result['messages']:

            # check the type of message with class
            if (isinstance(message, HumanMessage)):
                # human message which asked by the user
                query_result['query'] = message.content

            elif (isinstance(message, AIMessage)):
                if message.content:
                    query_result['final_answer'] = message.content

                else:
                    # check whether reasoning trace key is present
                    if message.additional_kwargs and "reasoning_content" in message.additional_kwargs:
                        query_result['reasoning_trace'].append(message.additional_kwargs["reasoning_content"])
                    # metadata message
                    query_result['response_metadata'] = message.response_metadata
                    query_result['tool_calls'] = message.tool_calls
                    
            elif (isinstance(message, ToolMessage)):
                # get the retrieved documents from tool message
                documents = message.content
                if documents:
                    # split the content document wise; every document starts with "Document X (Category: ..."
                    doc_splits = documents.split("Document ")
                    for doc_split in doc_splits[1:]:  # skip the first split as it will be empty
                        header, *content_lines = doc_split.splitlines()
                        header_parts = header.split("):", 1)
                        doc_info = header_parts[0] + ")"
                        doc_content = "\n".join(content_lines).strip()

                        query_result['retrieved_documents'].append({
                            "info": doc_info.strip(),
                            "content": doc_content
                        })

        # save output to a file
        output_filename = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = self.QUERIES_LOG_DIR / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(query_result, f, indent=2)

        return query_result

    # --------------------------------------------------
    # Main Execution
    # --------------------------------------------------
    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Run RAG + elicited reasoning.

        Returns a research-ready result object.
        """

        print("=" * 70)
        print(f"QUERY: {user_query}")
        print("=" * 70)

        try:
            # Invoke agent
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": user_query}]}
            )
            
            return self._parse_response(result)

        except Exception as e:
            # show exact error
            from traceback import format_exc
            # print("Error during agent execution:")
            print(format_exc())
            return {
                "query": user_query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
