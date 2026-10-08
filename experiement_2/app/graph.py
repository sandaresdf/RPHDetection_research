from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.agent import AgentState, agent_node, build_llm
from app.tools import web_search, send_email

def should_continue(state: AgentState):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_graph():

    llm = build_llm()
    tools_node = ToolNode([web_search, send_email])

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", lambda s: agent_node(s, llm))
    workflow.add_node("tools", tools_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    return workflow.compile()
