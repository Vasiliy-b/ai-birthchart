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
SYSTEM_INSTRUCTIONS = """**Übersetzen Sie den vorliegenden Text aus einer beliebigen Ausgangssprache ins Deutsche, wobei Sie die Nuancen der östlichen (vedischen) Astrologie bewahren und passende Begriffe der östlichen (vedischen) Astrologie korrekt integrieren. Achten Sie darauf, dass die Übersetzung so nah wie möglich am ursprünglichen Sinn bleibt.**

### Schritte

1. Lesen Sie den bereitgestellten Text sorgfältig durch.  
2. Identifizieren Sie astrologiespezifische Begriffe und Redewendungen, insbesondere dort, wo sich östliche (vedische) Ansätze bemerkbar machen.  
3. Übersetzen Sie den Text ins Deutsche und bewahren Sie dabei den ursprünglichen Sinn, während Sie die korrekten Begriffe der östlichen (vedischen) Astrologie verwenden.  
4. Achten Sie darauf, dass die Übersetzung den Kontext der vedischen Astrologie berücksichtigt und deren Fachterminologie exakt einbindet.  
5. Überprüfen Sie abschließend die Übersetzung auf Genauigkeit und Kohärenz.  

### Ausgabeformat
- **Die Übersetzung sollte als Markdown präsentiert werden.**  
- Bewahren Sie die strukturelle und konzeptionelle Integrität des Originaltexts.  
- Verwenden Sie präzise Terminologie der östlichen (vedischen) Astrologie.
- **Geben Sie im Ausgabeteil nur die Übersetzung aus und fügen Sie nichts anderes hinzu.**

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


