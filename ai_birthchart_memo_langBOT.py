import os
import uuid
import json
import logging
from datetime import datetime
from typing import Annotated, Dict, List, Any, Union

import getpass
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store import InMemoryStore

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    memory_context: Dict[str, Any]

class MemoryManager:
    def __init__(self, embedding_model, llm):
        """
        Initialize memory management system
        
        :param embedding_model: Embedding model for semantic search
        :param llm: Language model for summarization
        """
        self.store = InMemoryStore(
            index={
                "embed": embedding_model,
                "dims": 1024  # Adjust based on embedding model
            }
        )
        self.embedding_model = embedding_model
        self.llm = llm

    def summarize_memories(self, memories: List[Dict]) -> str:
        """
        Summarize multiple memories into a concise overview
        
        :param memories: List of memory dictionaries
        :return: Summarized memory context
        """
        try:
            memory_texts = [json.dumps(memory) for memory in memories]
            
            summarization_prompt = f"""
            Professionally summarize these conversation memories, 
            extracting key themes, important context, and recurring patterns:
            
            {memory_texts}
            
            Provide a structured, concise summary that captures 
            the essence of the interactions.
            """
            
            summary = self.llm.invoke([
                SystemMessage(content="You are an expert memory summarization assistant."),
                HumanMessage(content=summarization_prompt)
            ])
            
            return summary.content
        except Exception as e:
            logger.error(f"Memory summarization failed: {e}")
            return "Unable to summarize memories."

    def store_memory(self, user_id: str, memory_data: Dict):
        """
        Store a memory with semantic embedding
        
        :param user_id: Unique identifier for the user
        :param memory_data: Memory content to store
        """
        try:
            namespace = ("user_memories", user_id)
            memory_id = str(uuid.uuid4())
            
            self.store.put(
                namespace, 
                memory_id, 
                {
                    "data": memory_data,
                    "timestamp": datetime.now().isoformat()
                }
            )
            logger.info(f"Memory stored for user {user_id}")
        except Exception as e:
            logger.error(f"Memory storage failed: {e}")

    def retrieve_memories(self, user_id: str, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve most relevant memories for a user
        
        :param user_id: Unique identifier for the user
        :param query: Semantic search query
        :param top_k: Number of memories to retrieve
        :return: List of retrieved memories
        """
        try:
            namespace = ("user_memories", user_id)
            
            memories = self.store.search(
                namespace, 
                query=query, 
                k=top_k
            )
            
            return [memory.value["data"] for memory in memories]
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []

def priestess(state: State, config: RunnableConfig):
    """
    Agent node that incorporates memory context
    
    :param state: Current state of the conversation
    :param config: Configuration for the current invocation
    :return: Updated state with memory context
    """
    user_id = config["configurable"]["user_id"]
    
    # Retrieve relevant memories
    previous_memories = memory_manager.retrieve_memories(
        user_id, 
        query=state["messages"][-1].content
    )
    
    # Summarize memories if too many
    if len(previous_memories) > 3:
        memory_summary = memory_manager.summarize_memories(previous_memories)
    else:
        memory_summary = json.dumps(previous_memories)
    
    # Modify system message to include memory context
    system_message = SystemMessage(
        content=f"""
        {SYSTEM_INSTRUCTIONS}
        
        Relevant User Memory Context:
        {memory_summary}
        """
    )
    
    # Prepare messages with memory context
    formatted_messages = [system_message] + state["messages"]
    
    # Generate response
    response = llm.invoke(formatted_messages)
    
    # Store the new interaction as a memory
    memory_manager.store_memory(
        user_id, 
        {
            "user_message": state["messages"][-1].content,
            "ai_response": response.content
        }
    )
    
    return {"messages": [response]}

# Initialize memory manager
embedding_model = GoogleGenerativeAIEmbeddings(model="text-multilingual-embedding-002")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21")
memory_manager = MemoryManager(embedding_model, llm)

# Define graph structure
graph_builder = StateGraph(State)
graph_builder.add_node("priestess", priestess)
graph_builder.add_edge(START, "priestess")
graph_builder.add_edge("priestess", END)

# Compile graph with memory persistence
graph = graph_builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["priestess"]
)

# # Example usage
#
# user_id = "unique_user_id"  # Replace with actual user ID from backend
# config = {"configurable": {"user_id": user_id}}
# result = graph.invoke(
#     {"messages": [HumanMessage(content="Hello")]}, 
#     config=config
# )
