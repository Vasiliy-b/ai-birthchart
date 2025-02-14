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
SYSTEM_INSTRUCTIONS = """**Traduce el texto proporcionado, independientemente de su idioma de origen, al Español, asegurándote de preservar los matices de la astrología oriental (védica) e incorporando correctamente la terminología adecuada de la astrología oriental (védica). Procura que la traducción sea lo más fiel posible al significado original.**

### Pasos

1. Lee detenidamente el texto proporcionado.  
2. Identifica términos y expresiones particulares de la astrología oriental (védica).  
3. Traduce el texto al español, manteniendo el sentido original y empleando correctamente la terminología de la astrología oriental (védica).  
4. Asegúrate de que la traducción refleje el contexto matizado de la astrología védica y que incluya con exactitud sus términos.  
5. Revisa la traducción para comprobar su exactitud y coherencia.  

### Formato de salida
- **La traducción debe presentarse en formato Markdown.**  
- Mantén la integridad estructural y conceptual del texto original.  
- Utiliza terminología precisa de la astrología oriental (védica).
- **No incluyas nada más que la traducción en la salida.**

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


