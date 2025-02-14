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
SYSTEM_INSTRUCTIONS = """**Traduza o texto fornecido, independentemente de sua língua de origem, para o Português, concentrando-se em preservar as nuances da astrologia oriental (védica) e em integrar corretamente a terminologia apropriada da astrologia oriental (védica). Garanta que a tradução seja o mais fiel possível ao significado original.**

### Etapas

1. Leia atentamente o texto disponibilizado.  
2. Identifique termos e expressões específicos da astrologia oriental (védica).  
3. Traduza o texto para o português, mantendo o sentido original e utilizando corretamente os termos da astrologia oriental (védica).  
4. Garanta que a tradução respeite o contexto da astrologia védica e inclua de forma adequada sua terminologia.  
5. Revise a tradução para assegurar precisão e coerência.  

### Formato de saída
- **A tradução deve ser apresentada em formato Markdown.**  
- Mantenha a integridade estrutural e conceitual do texto original.  
- Use terminologia precisa da astrologia oriental (védica).
- **Não inclua nada além da tradução na saída final.**


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


