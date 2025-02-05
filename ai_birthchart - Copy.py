from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Dict, Any
import os

# 1. Define the State (EXACTLY as in example for now)
from langgraph.graph import MessagesState
class ConversationState(MessagesState): # Rename to ConversationState for consistency
    summary: str # We are not using summary yet, but keeping it for now

# 2. Define Nodes (Functions) - Simplified for basic communication
def call_gemini_api(state: ConversationState): # Keep 'state: ConversationState'
    """Node to call the Gemini API."""
    print("\n--- Gemini API Call Node ---")
    messages = state['messages'] # Get messages directly from state

    # Initialize Gemini Model
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", google_api_key=gemini_api_key)

    # Call the model
    response = model.invoke(messages)
    gemini_response_text = response.content
    print(f"Gemini Response: {gemini_response_text}")

    return {"messages": [AIMessage(content=gemini_response_text)]} # Return AIMessage

def get_user_message(state: ConversationState): # Dummy node for now, not really used in direct invocation
    # This node will not be used in direct invocation test, but keep it for graph structure
    pass # Placeholder - will replace with input() later


# 3. Build the LangGraph - Minimal Linear Graph
workflow = StateGraph(ConversationState) # Use example-like State class

# Add nodes - Keep names consistent
workflow.add_node("user_input_node", get_user_message) # Keep user_input_node, even if not used directly now
workflow.add_node("gemini_api_call", call_gemini_api) # Keep gemini_api_call

# Set edges - Simple linear flow
workflow.add_edge("user_input_node", "gemini_api_call")
workflow.set_entry_point("user_input_node") # Entry point, though we'll invoke differently
workflow.add_edge("gemini_api_call", END)

# 4. Compile the Graph
graph = workflow.compile()

# # 5. Direct Invocation for Testing - HARDCODED INPUT - Important Change
# if __name__ == "__main__":
#     print("--- Starting LangGraph Workflow (Direct Invocation Test) ---")
#     # Directly invoke with a HumanMessage - no user input yet
#     inputs = {"messages": [HumanMessage(content="Hello, Gemini! Are you there?")]} # Hardcoded input for now
#     result = graph.invoke(inputs)
#     print("\n--- Workflow Finished ---")
#     print("\nFinal Result State:", result)
#     print("\nGemini's Response:")
#     print(result['messages'][-1].content) # Print Gemini's response content