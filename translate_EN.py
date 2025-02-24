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
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21", google_api_key=gemini_api_key)

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] # Use BaseMessage for message flexibility

graph_builder = StateGraph(State)

llm = model

# System Instruction Prompt (Customize this as needed for tone/persona)
SYSTEM_INSTRUCTIONS = """
**Translate the provided text (from any source language) into English**, paying special attention to preserving the nuances of Eastern (Vedic) astrology and the correct usage of Eastern (Vedic) astrology terms. Make sure the translation is as close as possible to the original meaning.

### Steps

1. Carefully read the original text.  
2. Identify the astrological terms and expressions characteristic of Eastern (Vedic) astrology.  
3. Translate the text into English, preserving the original meaning and correctly using the corresponding Eastern (Vedic) astrology terminology.  
4. Make sure the translation accounts for the subtleties of Vedic astrology and accurately conveys its concepts.  
5. Check the translation for accuracy and coherence.

### Output Format

- All textual data, regardless of length (from a single word to an entire text), must be translated.  
- Preserve the structure and conceptual integrity of the original text.  
- Use accurate Eastern (Vedic) astrology terms.  
- **Do not include anything in the final answer except the translation itself.**

"""


def translator(state: State):
    # 1. Extract User Message
    user_messages = state.get("messages", [])

    
    # 3. Create system message with just the instructions 
    system_message = SYSTEM_INSTRUCTIONS

    # 4. Format messages for Gemini - keeping user conversation separate from system context
    formatted_messages = [system_message]  # System instructions first
    if user_messages:  # Then add any user messages
        formatted_messages.extend(user_messages)

    # 5. Invoke Gemini with the formatted messages
    response = llm.invoke(formatted_messages)

    # 6. Return AI Response
    return {"messages": [response]}


graph_builder.add_node("translator", translator)
graph_builder.add_edge(START, "translator")
graph_builder.add_edge("translator", END)

graph = graph_builder.compile()


