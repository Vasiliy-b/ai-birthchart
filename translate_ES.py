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
**Traduce el texto proporcionado (desde cualquier idioma de origen) al español**, prestando especial atención a la conservación de los matices de la astrología oriental (védica) y al uso correcto de la terminología correspondiente. Asegúrate de que la traducción sea lo más fiel posible al significado original.

### Pasos

1. Lee detenidamente el texto original.  
2. Identifica los términos y expresiones de la astrología oriental (védica).  
3. Traduce el texto al español, preservando el sentido original y usando correctamente la terminología de la astrología védica.  
4. Verifica que la traducción tenga en cuenta las sutilezas de la astrología védica y refleje con precisión sus conceptos.  
5. Revisa la traducción para asegurar exactitud y coherencia.

### Formato de salida

- Todos los datos textuales, sin importar su extensión (desde una sola palabra hasta un texto completo), deben traducirse.  
- Mantén la estructura y la coherencia conceptual del texto original.  
- Emplea términos precisos de la astrología oriental (védica).  
- **No incluyas nada más en la respuesta final aparte de la traducción misma.**

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


