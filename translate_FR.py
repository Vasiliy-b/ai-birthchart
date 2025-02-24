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
**Traduis le texte fourni (depuis n’importe quelle langue source) en français**, en veillant à préserver les nuances de l’astrologie orientale (védique) et à employer correctement les termes correspondants. Assure-toi que la traduction reste la plus fidèle possible au sens original.

### Étapes

1. Lis attentivement le texte original.  
2. Identifie les termes et expressions caractéristiques de l’astrologie orientale (védique).  
3. Traduis le texte en français en conservant le sens initial et en utilisant correctement la terminologie de l’astrologie védique.  
4. Assure-toi que la traduction tient compte des subtilités de l’astrologie védique et reflète avec précision ses concepts.  
5. Vérifie la traduction pour garantir sa précision et sa cohérence.

### Format de sortie

- Tous les contenus textuels, quelle que soit leur longueur (d’un seul mot à un texte entier), doivent être traduits.  
- Préserve la structure et l’intégrité conceptuelle du texte original.  
- Utilise des termes précis de l’astrologie orientale (védique).  
- **N’inclus rien d’autre dans la réponse finale en dehors de la traduction elle-même.**


"""


def translator(state: State):
    # 1. Extract User Message and Astro Data from State
    user_messages = state.get("messages", [])

    
    # 3. Create system message with just the instructions and astro data
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


