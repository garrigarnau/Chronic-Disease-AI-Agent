from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from agents.tools import query_db_metadata
from agents.tools import vector_search_chronic_diseases

# 1. Define the State
# This keeps track of the conversation history and the agent's thought process
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Initialize the Model with Tools
# We "bind" the tools so the LLM knows how to call them
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [vector_search_chronic_diseases, query_db_metadata]
llm_with_tools = llm.bind_tools(tools)

# 3. Define the Logic Nodes
def call_model(state: AgentState):
    """The 'Brain' node: Decides what to do next."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Router: Determines if we should call a tool or finish."""
    last_message = state['messages'][-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"

# 4. Build the Graph
workflow = StateGraph(AgentState)

# Add our two main nodes
workflow.add_node("agent", call_model)
workflow.add_node("action", ToolNode(tools))

# Set the entry point
workflow.set_entry_point("agent")

# Add conditional edges (The Decision Loop)
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)

# Connect actions back to the agent for "reasoning" over results
workflow.add_edge("action", "agent")

# Compile the brain
app = workflow.compile()