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

Full Table Schema:
- year_start             : The year data collection began (integer, e.g. 2020).
- location_desc          : State or region (e.g. 'Alabama', 'National').
- topic                  : Disease category (e.g. 'Diabetes', 'Asthma').
- question               : The specific metric/indicator being measured.
- data_value             : The numerical result (float).
- data_value_unit        : Unit of the value (e.g. '%', 'Number', 'cases per 100,000').
- data_value_type        : How the value was calculated — IMPORTANT for comparisons.
                           Common values: 'Crude Prevalence', 'Age-adjusted Prevalence',
                           'Age-adjusted Rate', 'Number', 'Crude Rate'.
                           Always filter by this when comparing across populations.
- stratification_category1: The demographic dimension (e.g. 'Overall', 'Gender', 'Race/Ethnicity', 'Age').
- stratification1        : The specific demographic value (e.g. 'Overall', 'Male', 'Female',
                           'White', 'Hispanic', 'Age >=65').
- low_confidence_limit   : Lower bound of the 95% confidence interval.
- high_confidence_limit  : Upper bound of the 95% confidence interval.

TIP: For population-level comparisons always use stratification_category1 = 'Overall'
unless the user explicitly asks for breakdowns by gender, race, or age.

TOOL USAGE — TWO-STEP PATTERN:

Step 1 — Concept discovery (vector_search_chronic_diseases):
  Call this whenever the user refers to a vague concept (e.g. 'respiratory issues',
  'heart disease', 'substance abuse').
  - If the question includes a year or location, pass them as the 'year' and
    'location' arguments so the discovery is already scoped correctly.
  - It returns the EXACT topic and question strings that exist in the database.

Step 2 — Precise retrieval (query_db_metadata):
  ALWAYS follow up with SQL when the user asked about a specific year, location,
  or demographic. Use the discovered topic/question names from Step 1.
  Example:
    SELECT location_desc, topic, data_value_type, AVG(data_value)
    FROM chronic_disease_indicators
    WHERE topic IN ('Asthma', 'Chronic Obstructive Pulmonary Disease')
      AND year_start = 2021
      AND location_desc = 'Colorado'
      AND stratification_category1 = 'Overall'
    GROUP BY location_desc, topic, data_value_type
      AND stratification_category1 = 'Overall'
    GROUP BY location_desc, topic, data_value_type

WHEN TO USE EACH PATTERN:
- Vague concept + no filters  → vector_search only
- Vague concept + filters     → vector_search THEN query_db_metadata
- Precise column value + filters → query_db_metadata only

OTHER GUIDELINES:
- If a query fails, analyze the error and try a different SQL approach.
- Never guess topic or question names — always discover them via vector_search first.
- Always specify data_value_type in GROUP BY to avoid mixing incompatible metrics."""

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