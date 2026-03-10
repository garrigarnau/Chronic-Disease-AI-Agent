import os
import sys
from pathlib import Path

# Ensure the project root is on the path so sibling packages resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from agents.orchestrator import app
from database import conn, cur

load_dotenv(Path(__file__).resolve().parent.parent / "config" / ".env")



# 1. Define the System Instructions
SYSTEM_PROMPT = """You are a Chronic Disease Data Assistant. 
You have access to a PostgreSQL table: 'chronic_disease_indicators'.

Table Schema:
- year_start: The year the data collection began.
- location_desc: The State or Region (e.g., 'Alabama', 'California').
- topic: The disease category (e.g., 'Diabetes', 'Cardiovascular Disease').
- question: The specific metric being measured.
- data_value: The numerical result.
- data_value_unit: Unit (%, cases, etc.).
- stratification1: Demographic info (Gender, Race, etc.).

GUIDELINES:
1. Use 'query_db_metadata' for specific stats, counts, or averages.
2. Use 'vector_search_chronic_diseases' if the user asks for general info or topics.
3. If a query fails, analyze the error and try a different SQL approach."""

def run_agent():
    print("--- 🩺 Chronic Disease AI Agent ---")
    print("(Type 'quit' to exit)\n")
    
    # Initialize session history with the system prompt
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        user_input = input("User: ")
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Closing database connections...")
            cur.close()
            conn.close()
            break

        # Add the user's question to the history
        messages.append(HumanMessage(content=user_input))
        
        # Execute the Graph
        # We use stream_mode="values" to get the full state updates
        inputs = {"messages": messages}
        
        print("\nAgent Thinking...")
        final_state = None
        
        # stream() lets you see the steps; app.invoke() would just give the final answer
        for output in app.stream(inputs, stream_mode="values"):
            final_state = output
            # Optional: Print tool calls here if you want to see the SQL being generated
        
        # Get the last message from the agent (the final response)
        response_msg = final_state["messages"][-1]
        print(f"\nAI: {response_msg.content}\n")
        
        # Update history with the agent's response for multi-turn conversation
        messages.append(response_msg)

if __name__ == "__main__":
    run_agent()