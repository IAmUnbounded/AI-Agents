"""
Example demonstrating the integration of language-specific agents with the code writing system.

This script shows how the system automatically selects the most appropriate programming
language for a task and uses specialized language-specific agents to generate code.
"""

from code_writing_agents import app, HumanMessage, AgentState

# Define a task that doesn't explicitly specify a language
task = """
Create a program that:
1. Reads data from a CSV file containing student grades (name, subject, score)
2. Calculates the average score for each student across all subjects
3. Identifies the top-performing student
4. Generates a summary report
"""

# Initialize the state with the task
initial_state = {
    "messages": [
        HumanMessage(content=task)
    ],
    "task": task,
    "plan": "",
    "code": "",
    "review": "",
    "test_results": "",
    "language": "",  # The language will be determined automatically
    "iteration_count": 0,
    "next": "orchestrator"
}

# Run the graph
print("\n===== LANGUAGE-SPECIFIC CODE WRITING SYSTEM =====\n")
print("Starting the multi-agent code writing process with language detection...\n")
print(f"Task: {task}\n")
final_state = app.invoke(initial_state, config={"configurable": {"thread_id": "language_integration_example"}})

# Print the final result
print("\n=== SELECTED LANGUAGE ===\n")
print(final_state["language"])

print("\n=== FINAL PLAN ===\n")
print(final_state["plan"])

print("\n=== FINAL CODE ===\n")
print(final_state["code"])

print("\n=== CODE REVIEW ===\n")
print(final_state["review"])

# Save the code to a file
if final_state["code"]:
    # Check if the code contains multiple files
    import re
    file_pattern = r"# FILENAME: ([^\n]+)"
    filenames = re.findall(file_pattern, final_state["code"])
    
    if filenames:
        print("\n=== SAVING MULTIPLE FILES ===\n")
        # Extract and save each file
        file_blocks_pattern = r"# FILENAME: ([^\n]+)\n\n(.*?)(?=\n\n# FILENAME:|$)"
        file_blocks = re.findall(file_blocks_pattern, final_state["code"], re.DOTALL)
        
        for filename, file_code in file_blocks:
            with open(filename, 'w') as f:
                f.write(file_code)
            print(f"Code saved to {filename}")
    else:
        # Determine appropriate file extension based on language
        language = final_state["language"]
        if language == "Python":
            file_extension = ".py"
            filename = "student_grades.py"
        elif language == "JavaScript":
            file_extension = ".js"
            filename = "student_grades.js"
        elif language == "C++":
            file_extension = ".cpp"
            filename = "student_grades.cpp"
        elif language == "HTML/CSS":
            file_extension = ".html"
            filename = "student_grades.html"
        else:
            file_extension = ".txt"
            filename = "student_grades.txt"
            
        # Save as a single file
        with open(filename, "w") as f:
            f.write(final_state["code"])
        print(f"\nCode saved to {filename}")
    
    # Create a sample CSV file for testing
    with open("students.csv", "w") as f:
        f.write("name,subject,score\n")
        f.write("Alice,Math,92\n")
        f.write("Alice,Science,88\n")
        f.write("Alice,English,95\n")
        f.write("Bob,Math,85\n")
        f.write("Bob,Science,92\n")
        f.write("Bob,English,78\n")
        f.write("Charlie,Math,90\n")
        f.write("Charlie,Science,93\n")
        f.write("Charlie,English,85\n")
    print("Sample data saved to students.csv")
    
    # Print instructions for running the code
    print("\nTo run the application:")
    if language == "Python":
        print(f"python {filename}")
    elif language == "JavaScript":
        print(f"node {filename}")
    elif language == "C++":
        print(f"g++ {filename} -o student_grades\n./student_grades")
    elif language == "HTML/CSS":
        print(f"Open {filename} in a web browser")
