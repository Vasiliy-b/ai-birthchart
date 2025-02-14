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
SYSTEM_INSTRUCTIONS = """**Traduisez le texte fourni, quelle que soit sa langue d’origine, en Français, en veillant à préserver les nuances de l’astrologie orientale (védique) tout en intégrant correctement la terminologie de l’astrologie orientale (védique). Assurez-vous que la traduction soit la plus fidèle possible au sens initial.**

### Étapes

1. Lisez attentivement le texte donné.  
2. Repérez les termes et expressions spécifiques à l’astrologie orientale (védique).  
3. Traduisez le texte en français en conservant le sens original et en utilisant correctement les termes de l’astrologie orientale (védique).  
4. Veillez à respecter le contexte nuancé de l’astrologie védique et à y inclure avec précision sa terminologie.  
5. Vérifiez la traduction pour en garantir l’exactitude et la cohérence.  

### Format de sortie
- **La traduction doit être présentée au format Markdown.**  
- Conservez l’intégrité structurelle et conceptuelle du texte original.  
- Employez une terminologie exacte de l’astrologie orientale (védique).
- **N’incluez rien d’autre que la traduction dans la sortie finale.**


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


