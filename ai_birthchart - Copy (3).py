import os
import getpass
import json
from typing import Annotated, Dict, List, Any, Union # More flexible typing
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage # Explicitly import message types


gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", google_api_key=gemini_api_key)

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] # Use BaseMessage for message flexibility
    astro_data: Dict[str, Any] # Add a field for your structured data

graph_builder = StateGraph(State)

llm = model

# System Instruction Prompt (Customize this as needed for tone/persona)
SYSTEM_INSTRUCTIONS = """You are a helpful astrological guide. 
You will receive user's astrological data and their questions. 
Use the data provided to answer their questions about their character, personality, or life aspects based on astrology. 
Maintain a helpful, insightful, and slightly mystical tone."""

def priestess(state: State):
    # 1. Extract User Message and Astro Data from State
    user_messages = state.get("messages", [])
    astro_data = state.get("astro_data", {}) # Get astro_data, default to empty dict if not present

    # 2. Format Input for Gemini: System Message + User Message
    formatted_messages: List[BaseMessage] = [
        SystemMessage(content=SYSTEM_INSTRUCTIONS + f"\n\nAstrological Data:\n{json.dumps(astro_data, indent=2)}") # Include data in SystemMessage
    ] 

    # Add user's messages (if any)
    formatted_messages.extend(user_messages)

    # 3. Invoke Gemini with the formatted messages
    response = llm.invoke(formatted_messages)

    # 4. Return AI Response
    return {"messages": [response]}


graph_builder.add_node("priestess", priestess)
graph_builder.add_edge(START, "priestess")
graph_builder.add_edge("priestess", END)

graph = graph_builder.compile()