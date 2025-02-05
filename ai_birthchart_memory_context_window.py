import os
import getpass
import json
from typing import Annotated, Dict, List, Any, Union
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from langgraph.checkpoint.memory import MemorySaver

gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", google_api_key=gemini_api_key)

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    astro_data: Dict[str, Any]

graph_builder = StateGraph(State)

llm = model

SYSTEM_INSTRUCTIONS = """You are a helpful astrological guide. 
You will receive user's astrological data and their questions. 
Use the data provided to answer their questions about their character, personality, or life aspects based on astrology. 
Maintain a helpful, insightful, and slightly mystical tone."""

# --- NEW in Step 4: Memory Buffer Size ---
MEMORY_WINDOW_SIZE = 5  # Keep last 5 turns (adjust as needed)
# ---------------------------------------

def priestess(state: State):
    # 1. Extract User Message and Astro Data from State
    user_messages = state.get("messages", [])
    astro_data = state.get("astro_data", {})

    # --- NEW in Step 4: Apply Memory Buffer ---
    # Keep only the last MEMORY_WINDOW_SIZE messages
    buffered_messages = user_messages[-MEMORY_WINDOW_SIZE:]
    # ---------------------------------------

    # 2. Format Input for Gemini: System Message + User Message (Buffered)
    formatted_messages: List[BaseMessage] = [
        SystemMessage(content=SYSTEM_INSTRUCTIONS + f"\n\nAstrological Data:\n{json.dumps(astro_data, indent=2)}")
    ]
    # Add BUFFERED user messages
    formatted_messages.extend(buffered_messages) # Use buffered_messages here

    # 3. Invoke Gemini with the formatted messages
    response = llm.invoke(formatted_messages)

    # 4. Return AI Response
    return {"messages": [response]}


graph_builder.add_node("priestess", priestess)
graph_builder.add_edge(START, "priestess")
graph_builder.add_edge("priestess", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

print("LangGraph with in-session memory buffer compiled and ready for LangGraph Studio!")