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
SYSTEM_INSTRUCTIONS = """**你要将所提供的文本（无论源语言是什么）翻译成简体中文**，并特别注意保留东方（吠陀）占星学的细微差别，同时正确使用与东方（吠陀）占星学相关的术语。确保翻译尽可能贴近原意。

### 步骤

1. 仔细阅读原文。  
2. 确认东方（吠陀）占星学中具有代表性的术语和表达方式。  
3. 将文本翻译成简体中文，保留原文含义并正确使用吠陀占星学相关术语。  
4. 确保翻译充分考虑吠陀占星学的细微之处，并准确反映其概念。  
5. 检查翻译的准确性和连贯性。

### 输出格式

- 所有文字内容，无论长度（从一个词到整篇文本），都需要翻译。  
- 保留原文的结构和概念完整性。  
- 使用准确的东方（吠陀）占星学术语。  
- **最终答案中只包含翻译内容，不要加入任何其他信息。**

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


