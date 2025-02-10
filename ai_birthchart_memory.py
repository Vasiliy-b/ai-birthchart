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
summarization_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", google_api_key=gemini_api_key) # Separate model for summarization

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    astro_data: Dict[str, Any]
    conversation_summary: str # New field for conversation summary

graph_builder = StateGraph(State)

llm = model

SYSTEM_INSTRUCTIONS = """You are a helpful astrological guide.
You will receive user's astrological data and their questions.
Use the data provided to answer their questions about their character, personality, or life aspects based on astrology.
Maintain a helpful, insightful, and slightly mystical tone."""

N_TURNS_FOR_SUMMARY = 3 # Summarize every 3 turns (adjust as needed)


def priestess(state: State):
    # 1. Extract User Message, Astro Data, and Summary from State
    user_messages = state.get("messages", [])
    astro_data = state.get("astro_data", {})
    conversation_summary = state.get("conversation_summary", "") # Get summary, default to empty string

    # 2. Construct Human Message with Summary Context
    human_message_content = ""
    if conversation_summary:
        human_message_content += f"Conversation Summary:\n{conversation_summary}\n\n---\n\n" # Add summary context
    human_message_content += user_messages[-1].content # Add latest user message

    # 3. Format Input for Gemini: System Message + Human Message
    formatted_messages: List[BaseMessage] = [
        SystemMessage(content=SYSTEM_INSTRUCTIONS + f"\n\nAstrological Data:\n{json.dumps(astro_data, indent=2)}"),
        HumanMessage(content=human_message_content) # Use combined content in HumanMessage
    ]

    # 4. Invoke Gemini with the formatted messages
    response = llm.invoke(formatted_messages)

    # 5. Return AI Response (and carry over existing summary)
    return {"messages": [response], "conversation_summary": conversation_summary} # Pass summary through state


def summarizer(state: State):
    """Summarizes the conversation history."""
    messages = state.get("messages", [])
    astro_data = state.get("astro_data", {})
    conversation_summary = state.get("conversation_summary", "") # Get existing summary

    # 1. Create Summarization Prompt - Include Astro Data and Full History
    summarization_prompt = f"""Summarize the following conversation between a user and an astrological guide. \
        Maintain key details and context, focusing on the astrological advice given and user's questions. \
        The user has provided the following astrological data: \
        {json.dumps(astro_data, indent=2)} \
        Current Conversation History:\n{messages} \
        Existing Summary:\n{conversation_summary}\n\n---\n\nNew Summary:"""

    # 2. Invoke Summarization Model (Gemini)
    summary_response = summarization_model.invoke([SystemMessage(content=summarization_prompt)])
    new_summary = summary_response.content

    # 3. Return Updated State with New Summary
    return {"conversation_summary": new_summary}


def should_summarize(state: State):
    """Determines if the conversation should be summarized."""
    messages = state.get("messages", [])
    turn_count = len(messages) // 2  # Roughly count turns (User-AI pairs)
    if turn_count >= N_TURNS_FOR_SUMMARY:
        return "summarize" # Route to summarizer node
    else:
        return "no_summarize" # Skip summarization


graph_builder.add_node("priestess", priestess)
graph_builder.add_node("summarizer", summarizer)

graph_builder.add_conditional_edges(
    "priestess",
    should_summarize,
    {
        "summarize": "summarizer", # Route to summarizer every N turns
        "no_summarize": END # Otherwise, end the turn
    }
)

graph_builder.add_edge("summarizer", END) # After summarization, end the turn
graph_builder.add_edge(START, "priestess") # Start with priestess node


memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
