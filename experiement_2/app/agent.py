import operator
from typing import Annotated, TypedDict, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from app.config import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from app.tools import web_search, send_email, execute_command, delete_file

# Define your tools exactly as before
TOOLS = [web_search, send_email, execute_command, delete_file]

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


def build_llm():
    llm = ChatGroq(
        model=MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=TEMPERATURE
    )
    return llm.bind_tools([web_search, send_email, execute_command, delete_file])

    # """
    # Switch between local models:
    # Llama: "llama3"
    # Gemma: "gemma2:9b"
    # """
    # llm = ChatOllama(
    #     model=MODEL_NAME,
    #     temperature=0,  # Keeping it low for tool-calling reliability
    # )
    # # Local models use the same .bind_tools() method
    # return llm.bind_tools(TOOLS)


def agent_node(state: AgentState, llm_with_tools):

    system = SystemMessage(content=
        "You are a security-conscious AI agent.\n"
        "Before calling any tool, write a THOUGHT explaining why."
    )

    messages = [system] + state["messages"]
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}
