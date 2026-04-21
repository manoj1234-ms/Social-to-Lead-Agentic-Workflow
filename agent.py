import os
import sys
import warnings
import logging

# Suppress LangGraph, HF, and Transformers deprecation warnings and download logs
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Mute the specific loggers causing the remaining output
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("langchain_huggingface").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "MOCK_KEY" # Just placing to avoid immediate crash for users who don't run it immediately

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Import Modular Project Files
from rag import search_knowledge_base
from tools import mock_lead_capture

def create_conversational_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    tools = [search_knowledge_base, mock_lead_capture]

    # Intent Detection & System Constraints
    system_prompt = """You are a conversational AI for AutoStream, a SaaS product that provides automated video editing tools for content creators.
Your task is to interact with users, categorize their intent, and advance them towards a lead conversion if they are highly interested.

You MUST correctly classify the user intent internally and start your response with:
[Intent: <Intent Category>]

where <Intent Category> is exactly one of:
- Casual greeting
- Product or pricing inquiry
- High-intent lead

Based on the intent:
1. Casual greeting: Greet them back politely.
2. Product or pricing inquiry: Use the `search_knowledge_base` tool to provide accurate information based on the knowledge base. Do not make up features or prices.
3. High-intent lead: When the user shows high intent (e.g. they say they want to sign up, subscribe, or try a plan), you MUST extract their Name, Email, and Creator Platform (like YouTube, Instagram, etc.). If any are missing, ask the user for them.
  **CRITICAL**: ONLY call the `mock_lead_capture` tool when you have collected ALL THREE values (name, email, platform). Do NOT call it prematurely.

Be helpful, engaging, and professional.
"""

    memory = MemorySaver()

    app = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        checkpointer=memory
    )
    return app

def chat_loop():
    print("AutoStream Conversational AI (type 'quit' to exit)")
    print("--------------------------------------------------")
    
    app = create_conversational_agent()
    config = {"configurable": {"thread_id": "user_session_1"}}
    
    while True:
        try:
            user_input = input("\nUser: ")
        except EOFError:
            break
            
        if user_input.lower() in ["quit", "exit", "q"]:
            break
            
        # Use invoke instead of stream for a perfectly clean terminal output!
        state = app.invoke({"messages": [("user", user_input)]}, config)
        msg = state['messages'][-1]
        
        # If content is a list of blocks, extract text
        if isinstance(msg.content, list):
            text = "".join([m.get("text", "") for m in msg.content if m.get("type") == "text"])
            print(f"\nAgent: {text}")
        else:
            print(f"\nAgent: {msg.content}")

if __name__ == "__main__":
    if os.environ.get("GOOGLE_API_KEY") == "MOCK_KEY":
        print("Warning: GOOGLE_API_KEY not set. Please set it in your environment or .env file before running.")
    else:
        chat_loop()
