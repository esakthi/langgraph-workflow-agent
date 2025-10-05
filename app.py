import os
import sqlite3
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from google_calendar import create_google_calendar_event
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import Tool

load_dotenv()

# 1. Define the tool for creating a calendar event
class CreateCalendarEvent(BaseModel):
    """Create a Google Calendar event."""
    summary: str = Field(description="A summary or title for the event.")
    start_datetime: str = Field(description="The start date and time for the event in ISO 8601 format (e.g., '2024-05-20T10:00:00').")
    end_datetime: str = Field(description="The end date and time for the event in ISO 8601 format (e.g., '2024-05-20T11:00:00').")

def create_event_tool(summary: str, start_datetime: str, end_datetime: str) -> str:
    """A helper function to create a Google Calendar event with the provided details."""
    event_details = {
        "summary": summary,
        "start": {"dateTime": start_datetime, "timeZone": "America/Los_Angeles"},
        "end": {"dateTime": end_datetime, "timeZone": "America/Los_Angeles"},
    }
    return create_google_calendar_event(event_details)

calendar_tool = Tool(
    name="create_calendar_event",
    func=create_event_tool,
    description="Creates a Google Calendar event with a summary, start time, and end time.",
    args_schema=CreateCalendarEvent,
)

tools = [calendar_tool]

# 2. Bind the tool to the LLM
llm = ChatGroq(model="llama3-70b-8192")
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 3. Define the nodes for the graph
def call_model(state):
    """Invokes the LLM with the current state."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

# 4. Define the conditional logic for routing
def should_continue(state):
    """Determines whether to continue with a tool call or end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue"
    return "end"

# 5. Build the graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_model)
graph.add_node("tool_node", tool_node)
graph.add_conditional_edges(
    "llm",
    should_continue,
    {"continue": "tool_node", "end": END},
)
graph.add_edge("tool_node", "llm")
graph.set_entry_point("llm")

# Set up memory
conn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn=conn)

# Compile the graph
app = graph.compile(checkpointer=memory)

def main():
    """Main function to run the workflow."""
    config = {"configurable": {"thread_id": "1"}}
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting...")
                break

            # The user input needs to be added to the state as a HumanMessage
            app.invoke({"messages": [HumanMessage(content=user_input)]}, config)

            # After the invoke, we can get the latest state and print the last message
            latest_state = app.get_state(config)
            latest_message = latest_state.messages[-1]

            # If the last message is a ToolMessage, it means the tool was called.
            if isinstance(latest_message, ToolMessage):
                print(f"Tool Output: {latest_message.content}")
            else:
                print(f"AI: {latest_message.content}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()