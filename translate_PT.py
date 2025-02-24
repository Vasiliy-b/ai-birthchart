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
**Traduz o texto fornecido (de qualquer idioma de origem) para o português**, prestando especial atenção à preservação das nuances da astrologia oriental (védica) e ao uso correto dos termos correspondentes. Garante que a tradução seja o mais fiel possível ao significado original.

### Etapas

1. Lê cuidadosamente o texto original.  
2. Identifica os termos e expressões característicos da astrologia oriental (védica).  
3. Traduz o texto para o português, mantendo o sentido original e utilizando corretamente a terminologia de astrologia védica.  
4. Assegura que a tradução leve em conta as sutilezas da astrologia védica e reflita com precisão seus conceitos.  
5. Verifica a tradução para garantir precisão e coerência.

### Formato de saída

- Todo o conteúdo textual, independentemente do tamanho (de uma única palavra a um texto completo), deve ser traduzido.  
- Mantém a estrutura e a integridade conceitual do texto original.  
- Usa termos exatos de astrologia oriental (védica).  
- **Não incluas nada além da tradução em tua resposta final.**

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


