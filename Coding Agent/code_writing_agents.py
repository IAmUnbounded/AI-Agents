"""
Multi-Agent System for Code Writing

This system consists of 5 agents:
1. Orchestrator Agent - Coordinates the workflow and manages communication
2. Planning Agent - Creates high-level plans and architecture
3. Coding Agent - Implements the actual code (now with language-specific agents)
4. Checking Agent - Reviews code for bugs and improvements
5. Testing Agent - Runs the code and identifies errors

The agents interact through a LangGraph workflow.
"""

import os
from typing import Dict, List, Annotated, TypedDict, Union, Literal
import json
import logging
import time
import sys
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import GoogleAPIError

from langchain_core.messages import (
    AIMessage, 
    HumanMessage, 
    SystemMessage,
    FunctionMessage,
    ChatMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Import the language-specific agents module
from language_specific_agents import (
    decide_language,
    language_specific_coding,
    get_language_coding_agent,
    get_file_extension,
    save_multiple_files
)

from langchain_core.messages import (
    AIMessage, 
    HumanMessage, 
    SystemMessage,
    FunctionMessage,
    ChatMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Import the language-specific agents module
from language_specific_agents import (
    decide_language,
    language_specific_coding,
    get_language_coding_agent,
    get_file_extension,
    save_multiple_files
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('code_agents.log')
    ]
)
logger = logging.getLogger('code_agents')

# Progress tracking function
def update_progress(stage, message):
    """Print progress update to console and log file"""
    progress_msg = f"[PROGRESS] {stage}: {message}"
    print(progress_msg)
    logger.info(progress_msg)
    # Force flush stdout to ensure progress is visible immediately
    sys.stdout.flush()

# Initialize LLM
update_progress("SETUP", "Initializing LLM with Gemini 2.0 Flash model")

# Configure the Gemini API
os.environ["GOOGLE_API_KEY"] = "your_api_key"  # Replace with your actual API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

try:
    # Use Gemini 2.0 Flash model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # Using Gemini 2.0 Flash model
        temperature=0.7,
        top_p=0.95,
        convert_system_message_to_human=True  # This helps with system messages
    )
    update_progress("SETUP", "Successfully initialized Gemini 2.0 Flash model")
except Exception as e:
    logger.error(f"Failed to initialize Gemini 2.0 Flash model: {str(e)}", exc_info=True)
    print(f"Error initializing Gemini 2.0 Flash: {str(e)}")
    sys.exit(1)

# Define tools that our agents can use
@tool
def save_code(code: str, filename: str) -> str:
    """Save the generated code to a file."""
    try:
        with open(filename, 'w') as f:
            f.write(code)
        return f"Code successfully saved to {filename}"
    except Exception as e:
        return f"Error saving code: {str(e)}"

@tool
def save_multiple_files(files_dict: Dict[str, str]) -> str:
    """Save multiple files at once.
    
    Args:
        files_dict: A dictionary mapping filenames to file contents
        
    Returns:
        A string describing the result of the save operation
    """
    results = []
    for filename, content in files_dict.items():
        try:
            with open(filename, 'w') as f:
                f.write(content)
            results.append(f"Successfully saved {filename}")
        except Exception as e:
            results.append(f"Error saving {filename}: {str(e)}")
    
    return "\n".join(results)

@tool
def read_file(filename: str) -> str:
    """Read the content of a file."""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def run_code(code: str, filename: str) -> str:
    """Run the code and capture any errors or output."""
    import tempfile
    import subprocess
    import os
    import sys
    
    try:
        # Save code to a temporary file
        with open(filename, 'w') as f:
            f.write(code)
        
        # Run the code and capture output/errors
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=10  # Timeout after 10 seconds to prevent infinite loops
        )
        
        # Prepare the result
        output = result.stdout
        errors = result.stderr
        return_code = result.returncode
        
        if return_code == 0 and not errors:
            return f"Code executed successfully.\nOutput:\n{output}"
        else:
            return f"Code execution failed with return code {return_code}.\nErrors:\n{errors}\nOutput:\n{output}"
    
    except subprocess.TimeoutExpired:
        return "Code execution timed out. It might contain an infinite loop or be too computationally intensive."
    except Exception as e:
        return f"Error running code: {str(e)}"

import re

def extract_code_from_response(content: str) -> str:
    """Extract code blocks from a response string.
    
    Args:
        content: The response content to extract code from
        
    Returns:
        The extracted code as a string
    """
    # Try to extract code blocks if they exist
    code_blocks = re.findall(r'```(?:python|html|css|javascript|js)?\s*(.*?)```', content, re.DOTALL)
    
    if code_blocks:
        # Join all code blocks if multiple are found
        code = "\n\n".join(code_blocks)
        return code
    else:
        # If no code blocks, try to find code between triple backticks without language
        code_blocks = re.findall(r'```\s*(.*?)```', content, re.DOTALL)
        if code_blocks:
            return "\n\n".join(code_blocks)
        
        # If still no code blocks, look for indented code (4 spaces or tab)
        indented_code = re.findall(r'(?:^|\n)(?:    |\t)(.*?)(?:$|\n\n)', content, re.DOTALL)
        if indented_code:
            return "\n".join(indented_code)
        
        # As a last resort, use the entire content
        return content

# Import string module for filename sanitization
import string

# Function to sanitize filenames
def sanitize_filename(filename):
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        A sanitized filename that is safe to use
    """
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
    """
    Ensure a file can be written to by checking and modifying permissions if needed.
    
    Args:
        filepath: Path to the file to check
        
    Returns:
        True if the file can be written, False otherwise
    """
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
            logger.error(f"Could not modify permissions for {filepath}: {str(e)}")
            return False
    # If file doesn't exist, check if directory is writable
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {str(e)}")
            return False
    return True

# Define the state for our graph
class AgentState(TypedDict):
    messages: List[Union[AIMessage, HumanMessage, SystemMessage, FunctionMessage]]
    task: str
    plan: str
    code: str
    review: str
    test_results: str
    language: str  # Added language field to track which language was selected
    iteration_count: int  # Add iteration counter to prevent infinite loops
    next: Literal["planning", "coding", "checking", "testing", "orchestrator", "end"]

# Define the agents

# 1. Orchestrator Agent
orchestrator_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are the Orchestrator Agent in a multi-agent system for writing code.
Your job is to:
1. Understand the user's requirements
2. Coordinate the work between the Planning, Coding, Checking, and Testing agents
3. Decide which agent should work next based on the current state
4. Summarize the final solution for the user

You should maintain a high-level view of the project and ensure all requirements are met.
"""),
    MessagesPlaceholder(variable_name="messages"),
])

# 2. Planning Agent
planning_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are the Planning Agent in a multi-agent system for writing code.
Your job is to:
1. Analyze the requirements provided by the Orchestrator
2. Create a detailed plan for implementing the code
3. Define the architecture, components, and interfaces
4. Consider edge cases and potential issues

Your output should be a structured plan that the Coding Agent can follow.
"""),
    MessagesPlaceholder(variable_name="messages"),
])

# 3. Coding Agent
coding_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are the Coding Agent in a multi-agent system for writing code.
Your job is to:
1. Implement code based on the plan provided by the Planning Agent
2. Follow best practices and coding standards
3. Include appropriate comments and documentation
4. Ensure the code is functional and efficient

Your output should be well-structured, clean code that addresses all requirements.
"""),
    MessagesPlaceholder(variable_name="messages"),
])

# 4. Checking Agent
checking_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are the Checking Agent in a multi-agent system for writing code.
Your job is to:
1. Review the code produced by the Coding Agent
2. Identify bugs, errors, or potential issues
3. Suggest improvements for efficiency, readability, or maintainability
4. Verify that the code meets all requirements from the original plan

Your output should be a detailed review with specific feedback and suggestions.
"""),
    MessagesPlaceholder(variable_name="messages"),
])

# 5. Testing Agent
testing_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are the Testing Agent in a multi-agent system for writing code.
Your job is to:
1. Run the code produced by the Coding Agent
2. Identify any runtime errors, bugs, or issues
3. Provide detailed feedback about what's not working
4. Suggest specific fixes for the identified issues
5. Test edge cases and verify functionality

Your output should be a detailed report of test results with specific recommendations for fixing any issues.
"""),
    MessagesPlaceholder(variable_name="messages"),
])

# Create the agent runnables
orchestrator_agent = orchestrator_prompt | llm
planning_agent = planning_prompt | llm
coding_agent = coding_prompt | llm
checking_agent = checking_prompt | llm
testing_agent = testing_prompt | llm

# Define the tools node
tools_node = ToolNode([save_code, read_file, run_code, save_multiple_files])

# Define the agent functions
def orchestrator(state: AgentState) -> AgentState:
    """Run the orchestrator agent on the current state."""
    logger.info("ORCHESTRATOR: Starting orchestrator agent")
    start_time = time.time()
    
    messages = state["messages"]
    task = state.get("task", "")
    plan = state.get("plan", "")
    code = state.get("code", "")
    review = state.get("review", "")
    test_results = state.get("test_results", "")
    language = state.get("language", "")
    
    # Get the current iteration count or initialize to 0
    iteration_count = state.get("iteration_count", 0) + 1
    logger.info(f"ORCHESTRATOR: Current iteration: {iteration_count}")
    
    # Check if we've exceeded the maximum iterations
    MAX_ITERATIONS = 10
    if iteration_count >= MAX_ITERATIONS:
        logger.warning(f"ORCHESTRATOR: Maximum iterations ({MAX_ITERATIONS}) reached, forcing workflow to end")
        return {"messages": messages, "task": task, 
                "plan": plan, "code": code, 
                "review": review, "test_results": test_results,
                "language": language,
                "iteration_count": iteration_count,
                "next": "end"}
    
    try:
        logger.info("ORCHESTRATOR: Invoking orchestrator LLM")
        # Make sure we have at least one message
        if not messages:
            messages = [HumanMessage(content="Please help with this task.")]
        
        # Determine language if not already set and we have a task
        if not language and task and not plan:
            # Try to determine language early
            language = decide_language(messages)
            logger.info(f"ORCHESTRATOR: Early language detection: {language}")
        
        # Add a specific instruction for the orchestrator to help it make better decisions
        orchestrator_instruction = HumanMessage(
            content=f"""Current state:
- Task: {task}
- Plan: {"Completed" if plan else "Not started"}
- Code: {"Completed" if code else "Not started"}
- Language: {language if language else "Not determined yet"}
- Review: {"Completed" if review else "Not started"}
- Test Results: {"Completed" if test_results else "Not started"}
- Iteration: {iteration_count}/{MAX_ITERATIONS}

Based on the current state, decide which agent should work next:
1. Planning agent - if we need to create or update the plan
2. Coding agent - if we have a plan but need to implement the code
3. Checking agent - if we have code that needs to be reviewed
4. Testing agent - if we have code that needs to be tested
5. End - if the task is complete

Your decision should be one of: "planning", "coding", "checking", "testing", or "complete".

{"Test results indicate issues that need to be fixed: " + test_results if test_results and ("error" in test_results.lower() or "bug" in test_results.lower() or "fix" in test_results.lower() or "issue" in test_results.lower() or "fail" in test_results.lower()) else ""}
"""
        )
        
        orchestrator_messages = messages + [orchestrator_instruction]
        
        logger.info("ORCHESTRATOR: Invoking orchestrator LLM")
        response = orchestrator_agent.invoke({"messages": orchestrator_messages})
        logger.info("ORCHESTRATOR: Received response from LLM")
        
        # Add the response to the messages
        messages.append(response)
        
        # Determine next step based on the orchestrator's decision
        content = response.content.lower()
        
        # Check if test results indicate issues that need fixing
        test_has_issues = test_results and ("error" in test_results.lower() or "bug" in test_results.lower() or 
                                           "fix" in test_results.lower() or "issue" in test_results.lower() or 
                                           "fail" in test_results.lower())
        
        # Force progression through the workflow if needed
        if test_has_issues and "planning" in content:
            # If tests show issues and orchestrator wants to go back to planning
            next_agent = "planning"
            logger.info("ORCHESTRATOR: Test results show issues, going back to planning")
        elif test_has_issues and "coding" in content:
            # If tests show issues and orchestrator wants to go back to coding
            next_agent = "coding"
            logger.info("ORCHESTRATOR: Test results show issues, going back to coding")
        elif plan and not code:
            # If we have a plan but no code, go to coding regardless of what the orchestrator says
            next_agent = "coding"
            logger.info("ORCHESTRATOR: Plan exists but no code, forcing progression to coding agent")
        elif code and not test_results:
            # If we have code but no test results, go to testing
            next_agent = "testing"
            logger.info("ORCHESTRATOR: Code exists but no test results, forcing progression to testing agent")
        elif code and test_results and not review:
            # If we have code and test results but no review, go to checking
            next_agent = "checking"
            logger.info("ORCHESTRATOR: Code and test results exist but no review, forcing progression to checking agent")
        elif "planning" in content or not plan:
            next_agent = "planning"
            logger.info("ORCHESTRATOR: Decided to proceed to planning")
        elif "coding" in content or not code:
            next_agent = "coding"
            logger.info("ORCHESTRATOR: Decided to proceed to coding")
        elif "testing" in content or not test_results:
            next_agent = "testing"
            logger.info("ORCHESTRATOR: Decided to proceed to testing")
        elif "checking" in content or not review:
            next_agent = "checking"
            logger.info("ORCHESTRATOR: Decided to proceed to checking")
        elif "complete" in content or "finished" in content or "end" in content:
            next_agent = "end"
            logger.info("ORCHESTRATOR: Task complete, ending workflow")
        else:
            # Default progression based on state
            if not plan:
                next_agent = "planning"
            elif not code:
                next_agent = "coding"
            elif not test_results:
                next_agent = "testing"
            elif not review:
                next_agent = "checking"
            else:
                next_agent = "end"
            logger.info(f"ORCHESTRATOR: No clear direction, defaulting to {next_agent} based on state")
        
        elapsed = time.time() - start_time
        logger.info(f"ORCHESTRATOR: Completed in {elapsed:.2f} seconds")
        
        return {"messages": messages, "task": task, 
                "plan": plan, "code": code, 
                "review": review, "test_results": test_results, 
                "language": language,
                "iteration_count": iteration_count,
                "next": next_agent}
    except GoogleAPIError as api_err:
        logger.error(f"Google API error in orchestrator: {str(api_err)}", exc_info=True)
        print(f"Google API error in orchestrator: {str(api_err)}")
        # Return to end if API error
        return {"messages": messages, "task": task, 
                "plan": plan, "code": code, 
                "review": review, "test_results": test_results, 
                "language": language,
                "iteration_count": iteration_count,
                "next": "end"}
    except Exception as e:
        logger.error(f"Error in orchestrator agent: {str(e)}", exc_info=True)
        print(f"Error in orchestrator: {str(e)}")
        # Return to end if too many errors
        return {"messages": messages, "task": task, 
                "plan": plan, "code": code, 
                "review": review, "test_results": test_results, 
                "language": language,
                "iteration_count": iteration_count,
                "next": "end"}

def planning(state: AgentState) -> AgentState:
    """Run the planning agent on the current state."""
    logger.info("PLANNING: Starting planning agent")
    start_time = time.time()
    
    messages = state["messages"]
    language = state.get("language", "")
    
    try:
        # Add a specific instruction for the planning agent
        planning_instruction = HumanMessage(
            content=f"Based on the task description, create a detailed plan for implementing the code{' in ' + language if language else ''}."
        )
        planning_messages = messages + [planning_instruction]
        
        logger.info("PLANNING: Invoking planning LLM")
        response = planning_agent.invoke({"messages": planning_messages})
        logger.info("PLANNING: Received response from LLM")
        
        # Add the response to the messages
        messages.append(response)
        
        # Extract the plan from the response
        plan = response.content
        logger.info("PLANNING: Plan created")
        logger.info(f"PLANNING: Plan length: {len(plan)} characters")
        
        # If language is not set yet, try to determine it from the plan
        if not language:
            language = decide_language(messages)
            logger.info(f"PLANNING: Detected language from plan: {language}")
        
        elapsed = time.time() - start_time
        logger.info(f"PLANNING: Completed in {elapsed:.2f} seconds")
        
        return {"messages": messages, "task": state.get("task", ""), 
                "plan": plan, "code": state.get("code", ""), 
                "review": state.get("review", ""), "test_results": state.get("test_results", ""),
                "language": language,
                "iteration_count": state.get("iteration_count", 0),
                "next": "orchestrator"}
    except GoogleAPIError as api_err:
        logger.error(f"Google API error in planning: {str(api_err)}", exc_info=True)
        print(f"Google API error in planning: {str(api_err)}")
        # Return to orchestrator to handle API error
        return {"messages": messages, "task": state.get("task", ""), 
                "plan": state.get("plan", ""), "code": state.get("code", ""), 
                "review": state.get("review", ""), "test_results": state.get("test_results", ""),
                "language": language,
                "iteration_count": state.get("iteration_count", 0),
                "next": "orchestrator"}
    except Exception as e:
        logger.error(f"Error in planning agent: {str(e)}", exc_info=True)
        print(f"Error in planning: {str(e)}")
        # Return to orchestrator to handle the error
        return {"messages": messages, "task": state.get("task", ""), 
                "plan": state.get("plan", ""), "code": state.get("code", ""), 
                "review": state.get("review", ""), "test_results": state.get("test_results", ""),
                "language": language,
                "iteration_count": state.get("iteration_count", 0),
                "next": "orchestrator"}

def coding(state: AgentState) -> AgentState:
    """Run the coding agent on the current state."""
    logger.info("CODING: Starting coding agent")
    start_time = time.time()
    
    messages = state["messages"]
    plan = state["plan"]
    task = state.get("task", "")
    
    try:
        # First, determine which language to use if not already decided
        language = state.get("language", "")
        if not language:
            language = decide_language(messages)
            logger.info(f"CODING: Selected language: {language}")
        else:
            logger.info(f"CODING: Using previously selected language: {language}")
        
        # Add a message to inform about language selection
        language_info_message = SystemMessage(
            content=f"The system has selected {language} as the most appropriate language for this task."
        )
        messages.append(language_info_message)
        
        # Use the language-specific coding function
        code, language = language_specific_coding(state, messages, plan)
        
        # Get the language-specific agent response to add to messages
        coding_instruction = HumanMessage(
            content=f"""Based on this plan:

{plan}

And this task:

{task}

Please implement the complete code in {language}."""
        )
        coding_messages = messages + [coding_instruction]
        coding_agent = get_language_coding_agent(language)
        response = coding_agent.invoke({"messages": coding_messages})
        
        # Add the response to the messages
        messages.append(response)
        
        elapsed = time.time() - start_time
        logger.info(f"CODING: Completed in {elapsed:.2f} seconds")
        
        return {"messages": messages, "task": state.get("task", ""), 
                "plan": plan, "code": code, 
                "review": state.get("review", ""), "test_results": state.get("test_results", ""),
                "language": language,  # Store the selected language in the state
                "iteration_count": state.get("iteration_count", 0),
                "next": "orchestrator"}
    except GoogleAPIError as api_err:
        logger.error(f"Google API error in coding: {str(api_err)}", exc_info=True)
        print(f"Google API error in coding: {str(api_err)}")
        # Return to orchestrator to handle API error
        return {"messages": messages, "task": state.get("task", ""), 
                "plan": plan, "code": state.get("code", ""), 
                "review": state.get("review", ""), "test_results": state.get("test_results", ""),
                "language": language,
                "iteration_count": state.get("iteration_count", 0),
                "next": "orchestrator"}
    except Exception as e:
        logger.error(f"Error in coding agent: {str(e)}", exc_info=True)
        print(f"Error in coding: {str(e)}")
        # Return to orchestrator to handle the error
        return {"messages": messages, "task": state.get("task", ""), 
                "plan": plan, "code": state.get("code", ""), 
                "review": state.get("review", ""), "test_results": state.get("test_results", ""),
                "language": language,
                "iteration_count": state.get("iteration_count", 0),
                "next": "orchestrator"}

def checking(state: AgentState) -> AgentState:
    """Run the checking agent on the current state."""
    logger.info("CHECKING: Starting checking agent")
    start_time = time.time()
    
    messages = state["messages"]
    plan = state["plan"]
    code = state["code"]
    language = state.get("language", "")
    
    try:
        # Add a specific instruction for the checking agent
        checking_instruction = HumanMessage(
            content=f"Review this {language} code:\n\n{code}\n\nBased on the plan:\n\n{plan}\n\nFocus on {language}-specific best practices and standards."
        )
        checking_messages = messages + [checking_instruction]
        
        logger.info("CHECKING: Invoking checking LLM")
        response = checking_agent.invoke({"messages": checking_messages})
        logger.info("CHECKING: Received response from LLM")
        
        # Add the response to the messages
        messages.append(response)
        
        # Extract the review from the response
        review = response.content
        logger.info("CHECKING: Code review complete")
        logger.info(f"CHECKING: Review length: {len(review)} characters")
        
        elapsed = time.time() - start_time
        logger.info(f"CHECKING: Completed in {elapsed:.2f} seconds")
        
        return {"messages": messages, "task": state.get("task", ""), 
                "plan": plan, "code": code, 
                "review": review, "test_results": state.get("test_results", ""),
                "language": language,
                "iteration_count": state.get("iteration_count", 0),
                "next": "orchestrator"}
    except GoogleAPIError as api_err:
        logger.error(f"Google API error in checking: {str(api_err)}", exc_info=True)
        print(f"Google API error in checking: {str(api_err)}")
        # Return to orchestrator to handle API error
        return {"messages": messages, "task": state.get("task", ""), 
                "plan": plan, "code": code, 
                "review": state.get("review", ""), "test_results": state.get("test_results", ""),
                "language": language,
                "iteration_count": state.get("iteration_count", 0),
                "next": "orchestrator"}
    except Exception as e:
        logger.error(f"Error in checking agent: {str(e)}", exc_info=True)
        print(f"Error in checking: {str(e)}")
        # Return to orchestrator to handle the error
        return {"messages": messages, "task": state.get("task", ""), 
                "plan": plan, "code": code, 
                "review": state.get("review", ""), "test_results": state.get("test_results", ""),
                "language": language,
                "iteration_count": state.get("iteration_count", 0),
                "next": "orchestrator"}

def testing(state: AgentState) -> AgentState:
    """Run the testing agent on the current state."""
    logger.info("TESTING: Starting testing agent")
    start_time = time.time()
    
    messages = state["messages"]
    plan = state["plan"]
    code = state["code"]
    language = state.get("language", "")
    
    try:
        # First, run the code to get test results
        logger.info("TESTING: Running code to identify issues")
        
        # Check if we have multiple files
        file_pattern = r"# FILENAME: ([^\n]+)"
        filenames = re.findall(file_pattern, code)
        
        # Get appropriate file extension based on language
        file_extension = get_file_extension(language) if language else ".py"
        
        if filenames:
            logger.info(f"TESTING: Detected multiple files: {filenames}")
            
            # Extract and save each file
            file_blocks_pattern = r"# FILENAME: ([^\n]+)\n\n(.*?)(?=\n\n# FILENAME:|$)"
            file_blocks = re.findall(file_blocks_pattern, code, re.DOTALL)
            
            files_dict = {}
            main_file = None
            main_content = None
            test_output = f"Multiple files detected for {language} project:\n"
            
            for filename, file_code in file_blocks:
                files_dict[filename] = file_code
                test_output += f"- {filename} ({len(file_code)} characters)\n"
                
                # Sanitize the filename
                sanitized_filename = sanitize_filename(filename)
                
                # Save each file
                try:
                    # Ensure the file can be written
                    if ensure_file_writable(sanitized_filename):
                        # Open with 'w+' mode to create or truncate the file
                        with open(sanitized_filename, 'w+') as f:
                            f.write(file_code)
                        
                        # Log if filename was changed
                        if sanitized_filename != filename:
                            logger.info(f"TESTING: Sanitized filename from '{filename}' to '{sanitized_filename}'")
                    else:
                        logger.warning(f"TESTING: Could not ensure write access to {sanitized_filename}")
                        test_output += f"Warning: Could not ensure write access to {sanitized_filename}\n"
                except OSError as e:
                    logger.error(f"TESTING: Error saving file {filename}: {str(e)}")
                    test_output += f"Error saving {filename}: {str(e)}\n"
                    continue
                
                # Determine the main file to run based on language
                if language == "Python" and sanitized_filename.endswith('.py') and (main_file is None or 'main' in sanitized_filename.lower()):
                    main_file = sanitized_filename
                    main_content = file_code
                elif language == "JavaScript" and sanitized_filename.endswith('.js') and (main_file is None or 'main' in sanitized_filename.lower() or 'index' in sanitized_filename.lower()):
                    main_file = sanitized_filename
                    main_content = file_code
                elif language == "C++" and sanitized_filename.endswith('.cpp') and (main_file is None or 'main' in sanitized_filename.lower()):
                    main_file = sanitized_filename
                    main_content = file_code
                elif language == "HTML/CSS" and sanitized_filename.endswith('.html'):
                    main_file = sanitized_filename
                    main_content = file_code
            
            # Run the main file based on language
            if main_file:
                if language == "Python" and main_file.endswith('.py'):
                    run_result = run_code(main_content, main_file)
                    test_output += f"\nExecution results for Python file {main_file}:\n{run_result}"
                elif language == "JavaScript" and main_file.endswith('.js'):
                    test_output += f"\nJavaScript file saved as {main_file}. To test: Run with Node.js or include in an HTML file."
                elif language == "C++" and main_file.endswith('.cpp'):
                    test_output += f"\nC++ file saved as {main_file}. To test: Compile with g++ and run the executable."
                elif language == "HTML/CSS" or main_file.endswith('.html'):
                    test_output += f"\nHTML file saved as {main_file}. To test: Open in a web browser."
                else:
                    test_output += f"\nFile saved as {main_file}. Manual testing required for this file type."
            else:
                test_output += f"\nNo runnable main file identified for {language}. Manual testing required."
        else:
            # Generate a filename based on the task and language
            task_words = state.get("task", "code").split()
            filename = "_".join([word.lower() for word in task_words[:3] if word.isalnum()]) + file_extension
            if not filename or filename == file_extension:
                filename = "test_code" + file_extension
            
            # Run the code and capture results based on language
            test_output = ""
            if language == "Python":
                test_output = run_code(code, filename)
            elif language == "JavaScript":
                # Save JavaScript file but don't execute
                with open(filename, 'w') as f:
                    f.write(code)
                test_output = f"JavaScript file saved as {filename}. To test: Run with Node.js or include in an HTML file."
            elif language == "C++":
                # Save C++ file but don't execute
                with open(filename, 'w') as f:
                    f.write(code)
                test_output = f"C++ file saved as {filename}. To test: Compile with g++ and run the executable."
            elif language == "HTML/CSS":
                # Save HTML file but don't execute
                with open(filename, 'w') as f:
                    f.write(code)
                test_output = f"HTML file saved as {filename}. To test: Open in a web browser."
            else:
                # For unknown languages, save the file but don't execute
                with open(filename, 'w') as f:
                    f.write(code)
                test_output = f"File saved as {filename}. For non-Python files, manual testing is required."
        
        logger.info(f"TESTING: Code handling complete. Output length: {len(test_output)} characters")
        
        # Add a specific instruction for the testing agent
        testing_instruction = HumanMessage(
            content=f"""Here is the code ({language if language else file_extension[1:]} file):

```
{code}
```

Here are the test results:

```
{test_output}
```

Based on the plan:

{plan}

Please analyze the code and test results. Identify any bugs, errors, or issues.
For {language if language else file_extension[1:]} code, focus on appropriate best practices and standards.
Provide specific recommendations for fixing any problems you find.
If the code appears correct, verify that it meets all requirements from the plan.
"""
        )
        
        testing_messages = messages + [testing_instruction]
        
        logger.info("TESTING: Invoking testing LLM")
        response = testing_agent.invoke({"messages": testing_messages})
        logger.info("TESTING: Received response from LLM")
        
        # Add the response to the messages
        messages.append(response)
        
        # Extract the test results from the response
        test_results = response.content
        logger.info("TESTING: Test analysis complete")
        logger.info(f"TESTING: Test results length: {len(test_results)} characters")
        
        # Determine if we need to go back to planning/coding based on test results
        needs_fixing = "error" in test_results.lower() or "bug" in test_results.lower() or "fix" in test_results.lower() or "issue" in test_results.lower() or "fail" in test_results.lower()
        
        next_step = "orchestrator"
        if needs_fixing:
            logger.info("TESTING: Issues detected, returning to orchestrator for fixes")
        else:
            logger.info("TESTING: No issues detected, continuing workflow")
        
        elapsed = time.time() - start_time
        logger.info(f"TESTING: Completed in {elapsed:.2f} seconds")
        
        return {"messages": messages, "task": state.get("task", ""), 
                "plan": plan, "code": code, 
                "review": state.get("review", ""), 
                "test_results": test_results, 
                "language": language,
                "iteration_count": state.get("iteration_count", 0),
                "next": next_step}
    except GoogleAPIError as api_err:
        logger.error(f"Google API error in testing: {str(api_err)}", exc_info=True)
        print(f"Google API error in testing: {str(api_err)}")
        # Return to orchestrator to handle API error
        return {"messages": messages, "task": state.get("task", ""), 
                "plan": plan, "code": code, 
                "review": state.get("review", ""), 
                "test_results": f"Error during testing: {str(api_err)}", 
                "language": language,
                "iteration_count": state.get("iteration_count", 0),
                "next": "orchestrator"}
    except Exception as e:
        logger.error(f"Error in testing agent: {str(e)}", exc_info=True)
        print(f"Error in testing: {str(e)}")
        # Return to orchestrator to handle the error
        return {"messages": messages, "task": state.get("task", ""), 
                "plan": plan, "code": code, 
                "review": state.get("review", ""), 
                "test_results": f"Error during testing: {str(e)}", 
                "language": language,
                "iteration_count": state.get("iteration_count", 0),
                "next": "orchestrator"}

def should_continue(state: AgentState) -> Literal["planning", "coding", "checking", "testing", "orchestrator", "end"]:
    """Determine which node to call next based on the state."""
    next_step = state["next"]
    logger.info(f"WORKFLOW: Moving to next step: {next_step}")
    return next_step

# Create the graph
logger.info("SETUP: Creating state graph")
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("orchestrator", orchestrator)
workflow.add_node("planning", planning)
workflow.add_node("coding", coding)
workflow.add_node("checking", checking)
workflow.add_node("testing", testing)
workflow.add_node("tools", tools_node)

# Add the edges - fixing the edge definitions
logger.info("SETUP: Adding conditional edges to graph")
workflow.add_conditional_edges(
    "orchestrator",
    should_continue,
    {
        "planning": "planning",
        "coding": "coding",
        "checking": "checking",
        "testing": "testing",
        "end": END
    }
)
workflow.add_edge("planning", "orchestrator")
workflow.add_edge("coding", "orchestrator")
workflow.add_edge("checking", "orchestrator")
workflow.add_edge("testing", "orchestrator")
workflow.add_edge("tools", "orchestrator")

# Set the entry point
workflow.set_entry_point("orchestrator")

# Compile the graph
logger.info("SETUP: Compiling graph")
app = workflow.compile()

# Initialize memory to persist state between graph runs
logger.info("SETUP: Initializing memory saver")
checkpointer = MemorySaver()

# Save the graph visualization
logger.info("SETUP: Skipping graph visualization to avoid network issues")
# Commenting out the graph visualization to avoid network errors
# graph_image = app.get_graph().draw_mermaid_png()
# with open('code_agents_graph.png', 'wb') as f:
#     f.write(graph_image)

# Example usage
if __name__ == "__main__":
    print("\n===== MULTI-AGENT CODE WRITING SYSTEM =====\n")
    print("Starting the multi-agent code writing process...\n")
    logger.info("MAIN: Starting code writing agents example")
    
    # Initialize the state with a user request
    initial_task = "Write a HTML code for a simple calculator"
    print(f"Task: {initial_task}\n")
    logger.info(f"MAIN: Initial task: {initial_task}")
    
    initial_state = {
        "messages": [
            HumanMessage(content=initial_task)
        ],
        "task": initial_task,
        "plan": "",
        "code": "",
        "review": "",
        "test_results": "",
        "language": "",
        "iteration_count": 0,  # Initialize iteration counter
        "next": "orchestrator"
    }
    
    logger.info("MAIN: Initial state created")
    
    try:
        # Run the graph
        print("Starting workflow execution...\n")
        logger.info("MAIN: Invoking the agent workflow")
        
        # Add progress tracking
        print("1. Orchestrator agent will analyze the task")
        print("2. Planning agent will create a detailed implementation plan")
        print("3. Coding agent will implement the code based on the plan")
        print("4. Testing agent will run the code and identify any issues")
        print("5. Checking agent will review the code for quality and correctness\n")
        
        print("Progress: Starting with Orchestrator...\n")
        
        # Add recursion limit configuration to prevent infinite loops
        final_state = app.invoke(
            initial_state, 
            config={
                "configurable": {"thread_id": "code_writing_example"},
                "recursion_limit": 50  # Increase the recursion limit to handle the workflow
            }
        )
        
        # Print the final result
        logger.info("MAIN: Workflow completed successfully")
        print("\n===== WORKFLOW COMPLETED SUCCESSFULLY =====\n")
        
        print("\n=== FINAL PLAN ===\n")
        print(final_state["plan"])
        
        print("\n=== FINAL CODE ===\n")
        print(final_state["code"])
        
        print("\n=== TEST RESULTS ===\n")
        print(final_state["test_results"])
        
        print("\n=== CODE REVIEW ===\n")
        print(final_state["review"])
        
        # Save the code to a file
        if final_state["code"]:
            # Check if we have multiple files
            file_pattern = r"# FILENAME: ([^\n]+)"
            filenames = re.findall(file_pattern, final_state["code"])
            
            if filenames:
                logger.info(f"MAIN: Detected multiple files: {filenames}")
                
                # Extract and save each file
                file_blocks_pattern = r"# FILENAME: ([^\n]+)\n\n(.*?)(?=\n\n# FILENAME:|$)"
                file_blocks = re.findall(file_blocks_pattern, final_state["code"], re.DOTALL)
                
                for filename, file_code in file_blocks:
                    sanitized_filename = sanitize_filename(filename)
                    # Ensure the file can be written
                    if ensure_file_writable(sanitized_filename):
                        with open(sanitized_filename, 'w+') as f:
                            f.write(file_code)
                        logger.info(f"MAIN: Code saved to {sanitized_filename}")
                        print(f"\nCode saved to {sanitized_filename}")
                    else:
                        logger.warning(f"MAIN: Could not ensure write access to {sanitized_filename}")
                        print(f"\nWarning: Could not ensure write access to {sanitized_filename}")
                
                print(f"\nMultiple files created. Check each file for specific instructions.")
            else:
                # Determine file extension based on code content
                code = final_state["code"]
                extension = ".py"  # Default extension
                
                # Check for HTML content
                if "<html" in code.lower() or "<!doctype html" in code.lower():
                    extension = ".html"
                # Check for JavaScript content
                elif "function" in code and ("document." in code or "window." in code):
                    extension = ".js"
                # Check for CSS content
                elif "{" in code and "}" in code and (":" in code) and (not "def " in code) and (not "class " in code):
                    extension = ".css"
                    
                filename = f"output{extension}"
                with open(filename, "w") as f:
                    f.write(final_state["code"])
                logger.info(f"MAIN: Code saved to {filename}")
                print(f"\nCode saved to {filename}")
                print(f"\nYou can run it with appropriate tools for {extension} files")
    except Exception as e:
        logger.error(f"ERROR: An error occurred: {str(e)}", exc_info=True)
        print(f"\n===== ERROR =====\n")
        print(f"An error occurred: {str(e)}")
        print("\nCheck the log file 'code_agents.log' for more details.")
