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
**Переведи предоставленный текст (с любого исходного языка) на русский язык, уделяя особое внимание сохранению нюансов восточной (ведической) астрологии и корректному использованию терминов восточной (ведической) астрологии. Убедись, что перевод максимально близок к исходному смыслу.**

### Шаги

1. Внимательно прочитай исходный текст.
2. Определи астрологические термины и выражения, характерные для восточной (ведической) астрологии.
3. Переведи текст на русский язык, сохраняя исходный смысл и корректно используя соответствующие термины восточной (ведической) астрологии.
4. Убедись, что перевод учитывает тонкости ведической астрологии и точно отражает её понятия.
5. Проверь перевод на точность и связность.

### Формат вывода

- Все текстовые данные, независимо от объёма (от одного слова до целого текста), подлежат переводу.
- Сохрани структуру и концептуальную целостность оригинального текста.
- Используй точные термины восточной (ведической) астрологии.
- **Не включай в окончательный ответ ничего, кроме самого перевода**
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


