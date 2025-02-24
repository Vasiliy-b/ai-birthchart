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
**Traduci il testo fornito (da qualsiasi lingua di origine) in italiano**, prestando particolare attenzione alla conservazione delle sfumature dell’astrologia orientale (vedica) e all’uso corretto della relativa terminologia. Assicurati che la traduzione sia il più possibile fedele al significato originario.

### Passaggi

1. Leggi attentamente il testo originale.  
2. Identifica i termini e le espressioni tipiche dell’astrologia orientale (vedica).  
3. Traduci il testo in italiano, mantenendo il senso originale e utilizzando correttamente la terminologia dell’astrologia vedica.  
4. Verifica che la traduzione tenga conto delle sfumature dell’astrologia vedica e ne rifletta accuratamente i concetti.  
5. Controlla la traduzione per verificarne l’accuratezza e la coerenza.

### Formato di output

- Tutto il contenuto testuale, indipendentemente dalla lunghezza (da una singola parola all’intero testo), deve essere tradotto.  
- Mantieni la struttura e l’integrità concettuale del testo originale.  
- Utilizza termini precisi dell’astrologia orientale (vedica).  
- **Non includere nulla nella risposta finale oltre alla traduzione stessa.**
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


