"""
Example usage of the multi-agent code writing system with language-specific agents.

This script demonstrates how the orchestrator, planning, coding, and checking agents
work together with the language decider to create code in the appropriate language.
"""

from code_writing_agents import app, HumanMessage, AgentState
import re
import os
import string

# Define a coding task
task = """
Create a modern website home page that:
1. Has a responsive design that works on desktop and mobile devices
2. Includes a navigation bar with links to Home, About, Services, and Contact sections
3. Features a hero section with a compelling headline and call-to-action button
4. Displays a grid of service offerings or product highlights
5. Includes a footer with contact information and social media links
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
    "language": "",  # The language will be determined by the language decider agent
    "iteration_count": 0,
    "next": "orchestrator"
}

# Run the graph
print("\n===== MULTI-AGENT CODE WRITING SYSTEM WITH LANGUAGE-SPECIFIC AGENTS =====\n")
print("Starting the multi-agent code writing process...\n")
print(f"Task: {task}\n")
final_state = app.invoke(initial_state, config={"configurable": {"thread_id": "website_homepage_example"}})

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
    # Determine appropriate file extension based on language
    language = final_state["language"]
    if language == "Python":
        file_extension = ".py"
        filename = "website.py"
    elif language == "JavaScript":
        file_extension = ".js"
        filename = "website.js"
    elif language == "HTML/CSS":
        file_extension = ".html"
        filename = "index.html"
    else:
        file_extension = ".txt"
        filename = "website.txt"
    
    # Function to sanitize filenames
    def sanitize_filename(filename):
        # Remove invalid characters
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        sanitized = ''.join(c for c in filename if c in valid_chars)
        # Remove any comment markers or other problematic patterns
        sanitized = sanitized.replace('/*', '').replace('*/', '')
        sanitized = sanitized.strip()
        # If filename becomes empty after sanitization, use a default
        if not sanitized:
            return "unnamed_file.txt"
        return sanitized
    
    # Function to ensure a file can be written (handles permissions and existence)
    def ensure_file_writable(filepath):
        # Check if file exists
        if os.path.exists(filepath):
            # Try to make it writable if it exists
            try:
                os.chmod(filepath, 0o666)  # Make file writable
                # On Windows, we may need to handle read-only attribute
                if os.name == 'nt':  # Windows
                    import stat
                    if not os.access(filepath, os.W_OK):
                        current_mode = os.stat(filepath).st_mode
                        os.chmod(filepath, current_mode | stat.S_IWRITE)
                return True
            except Exception as e:
                print(f"Warning: Could not modify permissions for {filepath}: {str(e)}")
                return False
        # If file doesn't exist, check if directory is writable
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception as e:
                print(f"Error creating directory {directory}: {str(e)}")
                return False
        return True
    
    # Check if the code contains multiple files
    file_pattern = r"# FILENAME: ([^\n]+)"
    import re
    filenames = re.findall(file_pattern, final_state["code"])
    
    if filenames:
        print("\n=== SAVING MULTIPLE FILES ===\n")
        # Extract and save each file
        file_blocks_pattern = r"# FILENAME: ([^\n]+)\n\n(.*?)(?=\n\n# FILENAME:|$)"
        file_blocks = re.findall(file_blocks_pattern, final_state["code"], re.DOTALL)
        
        for filename, file_code in file_blocks:
            # Sanitize the filename
            sanitized_filename = sanitize_filename(filename)
            try:
                # Ensure the file can be written
                if ensure_file_writable(sanitized_filename):
                    # Open with 'w+' mode to create or truncate the file
                    with open(sanitized_filename, 'w+') as f:
                        f.write(file_code)
                    print(f"Code saved to {sanitized_filename}" + 
                          (f" (original: {filename})" if sanitized_filename != filename else ""))
                else:
                    print(f"Warning: Could not ensure write access to {sanitized_filename}")
            except OSError as e:
                print(f"Error saving file {filename}: {str(e)}")
    else:
        # Save as a single file
        try:
            # Ensure the file can be written
            if ensure_file_writable(filename):
                # Open with 'w+' mode to create or truncate the file
                with open(filename, "w+") as f:
                    f.write(final_state["code"])
                print(f"\nCode saved to {filename}")
            else:
                print(f"Warning: Could not ensure write access to {filename}")
        except OSError as e:
            print(f"Error saving file {filename}: {str(e)}")
    
    # Create additional files based on language
    if language == "HTML/CSS":
        # Check if CSS file is already created in the multiple files
        css_created = any("styles.css" in fname for fname in filenames)
        js_created = any("script.js" in fname for fname in filenames)
        
        if not css_created:
            # Create an empty CSS file if not already included
            try:
                # Ensure the file can be written
                if ensure_file_writable("styles.css"):
                    with open("styles.css", "w+") as f:
                        f.write("/* CSS styles for the website */")
                    print("Created styles.css file")
                else:
                    print("Warning: Could not ensure write access to styles.css")
            except OSError as e:
                print(f"Error creating styles.css: {str(e)}")
        
        if not js_created:
            # Create an empty JavaScript file if not already included
            try:
                # Ensure the file can be written
                if ensure_file_writable("script.js"):
                    with open("script.js", "w+") as f:
                        f.write("// JavaScript for the website")
                    print("Created script.js file")
                else:
                    print("Warning: Could not ensure write access to script.js")
            except OSError as e:
                print(f"Error creating script.js: {str(e)}")

    # Print instructions for running the code
    print("\nTo view the website:")
    if language == "Python":
        print(f"Run: python {filename}")
        print("Then open a web browser and navigate to the URL shown in the terminal")
    elif language == "JavaScript" or language == "HTML/CSS":
        print(f"Open {filename} in a web browser")
