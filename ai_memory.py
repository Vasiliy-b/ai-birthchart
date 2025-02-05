import os
import getpass
import json
import uuid
from typing import Annotated, Dict, List, Any, Union
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from langgraph.checkpoint.memory import MemorySaver
# NEW: Import InMemoryStore for cross-session memory
from langgraph.store.memory import InMemoryStore
from langchain.prompts import ChatPromptTemplate # For summarization prompt

gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", google_api_key=gemini_api_key)
# NEW: Separate model for summarization (can be the same or a faster/cheaper model)
summarization_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", google_api_key=gemini_api_key) # Using the same model for now

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    astro_data: Dict[str, Any]
    # NEW: Add memory field to state
    user_memory: Dict[str, Any] # To store our JSON memory structure

graph_builder = StateGraph(State)

llm = model

SYSTEM_INSTRUCTIONS = """You are a helpful astrological guide. 
You will receive user's astrological data and their questions. 
Use the data provided to answer their questions about their character, personality, or life aspects based on astrology. 
Maintain a helpful, insightful, and slightly mystical tone."""

# --- NEW in Step 5: Summarization Prompt ---
SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Please provide a concise summary of the following conversation history, focusing on the key topics and user's goals. Aim for a summary that is informative but brief, capturing the essential context for future turns."),
    ("human", "{conversation_history}")
])
# ---------------------------------------

N_TURNS_FOR_SUMMARY = 5 # Summarize every 5 turns

def priestess(state: State):
    # 1. Extract User Message, Astro Data, and Memory from State
    user_messages = state.get("messages", [])
    astro_data = state.get("astro_data", {})
    user_memory = state.get("user_memory", {}) # NEW: Retrieve user_memory

    # 2. Format Input for Gemini: System Message + Human Message (with History)
    formatted_messages: List[BaseMessage] = [
        SystemMessage(content=SYSTEM_INSTRUCTIONS + f"\n\nAstrological Data:\n{json.dumps(astro_data, indent=2)}")
    ]
    # NEW: Prepend summarized history to HumanMessage content
    history_summary = user_memory.get("history", "") # Get history from memory
    current_user_message = user_messages[-1].content if user_messages else "" # Get current user message
    human_message_content = f"Conversation History Summary:\n{history_summary}\n\nUser's Current Input:\n{current_user_message}" # Concatenate
    formatted_messages.append(HumanMessage(content=human_message_content)) # Use concatenated content

    # 3. Invoke Gemini with the formatted messages
    response = llm.invoke(formatted_messages)

    # 4. Return AI Response (no change here)
    return {"messages": [response]}


def summarizer_node(state: State, store: InMemoryStore): # NEW: Add store argument
    user_messages = state.get("messages", [])
    user_memory = state.get("user_memory", {}) # NEW: Retrieve user_memory

    # --- NEW in Step 5: Summarization Logic ---
    if len(user_messages) >= N_TURNS_FOR_SUMMARY: # Check if it's time to summarize

        # Get conversation history for summarization prompt
        conversation_history_for_summary = "\\n".join([f"{msg.role}: {msg.content}" for msg in user_messages]) # Format history

        # Invoke summarization model
        summary_response = summarization_model.invoke(SUMMARIZATION_PROMPT.format_messages(conversation_history=conversation_history_for_summary))
        new_summary = summary_response.content

        # Update user_memory JSON (Option A - simple append to history)
        current_history = user_memory.get("history", "")
        updated_history = current_history + f"\n\n--- New Summary ---\n" + new_summary # Append summary
        user_memory["history"] = updated_history # Update history field

        # Store updated memory in InMemoryStore (cross-session memory - see Step 6 for persistence across sessions if needed)
        user_id = "user_1" #  For now, hardcode user_id (replace with actual user ID logic in real app)
        namespace = ("user_memory", user_id)
        store.put(namespace, "memory_key", user_memory) # Store the updated memory
        print("\n--- Conversation Summarized! ---\n") # Debug print

        # Replace full message history with just the summary (for now - can refine later)
        # state["messages"] = [SystemMessage(content="Conversation summarized. Summary: " + new_summary)] # Option to replace with just summary - not ideal

        # Keep full history for now, but we've updated user_memory with the summary
        pass # For now, keep full message history in state for priestess node to use in next turn

    # Even if not summarized, return the (potentially updated) state - state is updated in place, so no need to reassign unless you re-create it
    return {"user_memory": user_memory} # Return updated memory

graph_builder.add_node("priestess", priestess)
# NEW: Add summarizer node
graph_builder.add_node("summarizer", summarizer_node)

# --- NEW in Step 5: Graph Flow with Summarizer ---
graph_builder.add_edge(START, "priestess") # Start now goes to summarizer first
graph_builder.add_edge("priestess", "summarizer") # Summarizer then goes to priestess
graph_builder.add_edge("summarizer", "priestess")
graph_builder.add_edge("priestess", END) # End after summarizer
# ---------------------------------------------

memory = MemorySaver()
# NEW: Initialize InMemoryStore
in_memory_store = InMemoryStore()

# Compile the graph with both checkpointer and store
graph = graph_builder.compile(checkpointer=memory, store=in_memory_store) # Pass store to compile

