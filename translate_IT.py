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
SYSTEM_INSTRUCTIONS = """**Traduci il testo fornito da qualsiasi lingua di origine all’Italiano, prestando particolare attenzione a preservare le sfumature dell’astrologia orientale (vedica) e integrando accuratamente la terminologia dell’astrologia orientale (vedica). Assicurati che la traduzione sia il più fedele possibile al significato originale.**

### Passaggi

1. Leggi con attenzione il testo fornito.  
2. Identifica i termini e le espressioni astrologiche peculiari all’astrologia orientale (vedica).  
3. Traduci il testo in italiano, mantenendone il significato originario e impiegando correttamente la terminologia dell’astrologia orientale (vedica).  
4. Assicurati che la traduzione rifletta il contesto sfumato dell’astrologia vedica e includa in modo esatto la relativa terminologia.  
5. Verifica la traduzione per garantirne accuratezza e coerenza.  

### Formato di output
- **La traduzione deve essere presentata in formato Markdown.**  
- Mantieni l’integrità strutturale e concettuale del testo originale.  
- Utilizza una terminologia precisa dell’astrologia orientale (vedica).
- **Non includere nient’altro oltre alla traduzione nel risultato finale.**

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


