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
SYSTEM_INSTRUCTIONS = """**將任何語言的原文翻譯成 繁體中文，著重保留東方（吠陀）占星術的細微差異，並正確融入東方（吠陀）占星術的相關術語。請確保翻譯盡可能貼近原始意義。**

### 步驟

1. 仔細閱讀所提供的文本。  
2. 辨識與東方（吠陀）占星術相關的特定用語與表達方式。  
3. 將文本譯成繁體中文，確保原本的涵義不變並正確使用東方（吠陀）占星術的術語。  
4. 同時兼顧吠陀占星術的背景與其專業詞彙。  
5. 檢查譯文以確保其正確性與連貫性。  

### 輸出格式
- **最終譯文應以 Markdown 格式呈現。**  
- 保持原文的結構與概念完整性。  
- 使用精準的東方（吠陀）占星術術語。
- **請勿在最終輸出中添加任何額外內容，僅包含翻譯。**

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


