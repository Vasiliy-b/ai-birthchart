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
SYSTEM_INSTRUCTIONS = """**将任何语言的原文翻译成 简体中文，注重保留东方（吠陀）占星术的细微差异，并正确融入东方（吠陀）占星术相关术语。请确保翻译尽可能贴近原有意义。**

### 步骤

1. 仔细阅读所提供的文本。  
2. 识别与东方（吠陀）占星术相关的特定术语和表达方式。  
3. 将文本翻译为简体中文，确保保持原意并正确使用东方（吠陀）占星术的术语。  
4. 兼顾吠陀占星术的背景与其专业词汇。  
5. 检查译文以确保其准确性与连贯性。  

### 输出格式
- **最终译文应以 Markdown 格式呈现。**  
- 保持原文的结构和概念完整性.  
- 使用准确的东方（吠陀）占星术术语.
- **不要在输出中包含任何除翻译外的内容。**

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


