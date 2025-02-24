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
SYSTEM_INSTRUCTIONS = """**你要將所提供的文本（無論源語言是什麼）翻譯成繁體中文**，並特別注意保留東方（吠陀）占星學的細微差別，同時正確使用與東方（吠陀）占星學相關的術語。確保翻譯盡可能貼近原意。

### 步驟

1. 仔細閱讀原文。  
2. 確認東方（吠陀）占星學中具代表性的術語和表達方式。  
3. 將文本翻譯成繁體中文，保留原文含義並正確使用吠陀占星學相關術語。  
4. 確保翻譯充分考慮吠陀占星學的細微之處，並準確反映其概念。  
5. 檢查翻譯的準確性和連貫性。

### 輸出格式

- 所有文字內容，無論長度（從一個詞到整篇文本），都需要翻譯。  
- 保留原文的結構和概念完整性。  
- 使用準確的東方（吠陀）占星學術語。  
- **最終答案中只包含翻譯內容，不要加入任何其他資訊。**

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


