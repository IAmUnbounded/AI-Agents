"""
Example usage of the multi-agent code writing system.

This script demonstrates how the orchestrator, planning, coding, and checking agents
work together to create a more complex application.
"""

from code_writing_agents import app, HumanMessage, AgentState

# Define a more complex coding task
complex_task = """
Create a simple web API using FastAPI that:
1. Has an endpoint to add a new todo item
2. Has an endpoint to list all todo items
3. Has an endpoint to mark a todo item as complete
4. Stores todo items in memory (no database needed)
5. Includes proper error handling and validation
"""

# Initialize the state with the complex task
initial_state = {
    "messages": [
        HumanMessage(content=complex_task)
    ],
    "task": complex_task,
    "plan": "",
    "code": "",
    "review": "",
    "next": "orchestrator"
}

# Run the graph
print("Starting the multi-agent code writing process...")
final_state = app.invoke(initial_state, config={"configurable": {"thread_id": "fastapi_todo_example"}})

# Print the final result
print("\n=== FINAL PLAN ===\n")
print(final_state["plan"])

print("\n=== FINAL CODE ===\n")
print(final_state["code"])

print("\n=== CODE REVIEW ===\n")
print(final_state["review"])

# Save the code to a file
if final_state["code"]:
    with open("todo_api.py", "w") as f:
        f.write(final_state["code"])
    print("\nCode saved to todo_api.py")
    
    # Create a requirements.txt file for the FastAPI application
    with open("requirements.txt", "w") as f:
        f.write("fastapi==0.104.1\nuvicorn==0.23.2\npydantic==2.4.2")
    print("Requirements saved to requirements.txt")
    
    print("\nTo run the FastAPI application, execute:")
    print("pip install -r requirements.txt")
    print("uvicorn todo_api:app --reload")
