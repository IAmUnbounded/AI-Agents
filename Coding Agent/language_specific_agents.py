"""
Language-Specific Coding Agents

This module provides specialized coding agents for different programming languages:
1. Python Agent - Specialized for Python code generation
2. JavaScript Agent - Specialized for JavaScript code generation
3. C++ Agent - Specialized for C++ code generation
4. HTML/CSS Agent - Specialized for web frontend code generation
5. Language Decider Agent - Determines which language agent to use

These agents can be integrated with the main code_writing_agents system.
"""

import re
import logging
import time
from typing import Dict, List, Union, Literal, Tuple, Optional
import os

from langchain_core.messages import (
    AIMessage, 
    HumanMessage, 
    SystemMessage,
    FunctionMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# Set up logging
logger = logging.getLogger('language_agents')

# Configure the Gemini API with the same key used in code_writing_agents.py
api_key = "your_api_key"  # Replace with your actual API key
os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

# Initialize LLM (reusing from code_writing_agents)
llm = None  # Define llm variable with a default value
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        top_p=0.95,
        convert_system_message_to_human=True
    )
    logger.info("Successfully initialized Gemini model for language agents")
except Exception as e:
    logger.error(f"Failed to initialize Gemini model for language agents: {str(e)}", exc_info=True)
    print(f"Failed to initialize Gemini model for language agents: {str(e)}")
    # Create a dummy LLM that will raise an informative error if used
    class DummyLLM:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("LLM initialization failed. Please check your API key and credentials.")
    llm = DummyLLM()

# Helper function to extract code from responses (reused from code_writing_agents)
def extract_code_from_response(content: str) -> str:
    """Extract code blocks from a response string."""
    # Try to extract code blocks if they exist
    code_blocks = re.findall(r'```(?:python|html|css|javascript|js|cpp)?\s*(.*?)```', content, re.DOTALL)
    
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

# Helper function to save multiple files
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
            # Create directory if it doesn't exist
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(filename, 'w') as f:
                f.write(content)
            results.append(f"Successfully saved {filename}")
            logger.info(f"LANGUAGE_CODING: Saved file {filename} ({len(content)} characters)")
        except Exception as e:
            results.append(f"Error saving {filename}: {str(e)}")
            logger.error(f"LANGUAGE_CODING: Error saving file {filename}: {str(e)}")
    
    return "\n".join(results)

# 1. Python Coding Agent
python_coding_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are the Python Coding Agent, specialized in writing high-quality Python code.
Your job is to:
1. Implement code based on the plan provided
2. Follow Python best practices and PEP 8 standards
3. Use modern Python features (Python 3.8+)
4. Include appropriate docstrings and comments
5. Ensure the code is functional, efficient, and Pythonic
6. Leverage appropriate Python libraries and frameworks

Your output should be well-structured, clean Python code that addresses all requirements.
Always include all necessary imports and dependencies.

If the implementation requires multiple files, clearly indicate each file with a filename header:
```python
# FILENAME: example.py
# Code for example.py goes here
```

For packages, organize them properly with __init__.py files and appropriate module structure.
"""),
    MessagesPlaceholder(variable_name="messages"),
])

# 2. JavaScript Coding Agent
javascript_coding_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are the JavaScript Coding Agent, specialized in writing high-quality JavaScript code.
Your job is to:
1. Implement code based on the plan provided
2. Follow modern JavaScript best practices and standards
3. Use ES6+ features where appropriate
4. Include appropriate comments and JSDoc documentation
5. Ensure the code is functional, efficient, and follows JavaScript idioms
6. Leverage appropriate JavaScript libraries and frameworks when needed

Your output should be well-structured, clean JavaScript code that addresses all requirements.
Always include all necessary imports, requires, or script tags.

If the implementation requires multiple files, clearly indicate each file with a filename header:
```javascript
// FILENAME: example.js
// Code for example.js goes here
```

For web applications, create appropriate HTML, CSS, and JS files with proper separation of concerns.
"""),
    MessagesPlaceholder(variable_name="messages"),
])

# 3. C++ Coding Agent
cpp_coding_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are the C++ Coding Agent, specialized in writing high-quality C++ code.
Your job is to:
1. Implement code based on the plan provided
2. Follow modern C++ best practices and standards (C++11 or newer)
3. Use appropriate memory management techniques
4. Include appropriate comments and documentation
5. Ensure the code is functional, efficient, and follows C++ idioms
6. Consider performance and resource usage

Your output should be well-structured, clean C++ code that addresses all requirements.
Always include all necessary headers, namespaces, and dependencies.

If the implementation requires multiple files, clearly indicate each file with a filename header:
```cpp
// FILENAME: example.cpp
// Code for example.cpp goes here
```

For larger projects, properly separate header (.h) and implementation (.cpp) files.
"""),
    MessagesPlaceholder(variable_name="messages"),
])

# 4. HTML/CSS Coding Agent
html_css_coding_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are the HTML/CSS Coding Agent, specialized in writing high-quality web frontend code.
Your job is to:
1. Implement code based on the plan provided
2. Follow modern HTML5 and CSS3 best practices and standards
3. Create responsive and accessible web interfaces
4. Include appropriate comments and documentation
5. Ensure the code is functional, efficient, and follows web development best practices
6. Use semantic HTML and well-structured CSS

Your output should be well-structured, clean HTML and CSS code that addresses all requirements.
Always include all necessary meta tags, CSS links, or script tags.

If the implementation requires multiple files, clearly indicate each file with a filename header:
```html
<!-- FILENAME: index.html -->
<!-- HTML code goes here -->
```

```css
/* FILENAME: styles.css */
/* CSS code goes here */
```

```javascript
// FILENAME: script.js
// JavaScript code goes here
```

For web applications, create a proper directory structure with separate files for HTML, CSS, and JavaScript.
"""),
    MessagesPlaceholder(variable_name="messages"),
])

# 5. Language Decider Agent
language_decider_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are the Language Decider Agent in a multi-agent system for writing code.
Your job is to:
1. Analyze the requirements provided by the Orchestrator
2. Determine the most appropriate programming language for the task
3. Consider the nature of the problem, performance requirements, and use case
4. Choose from: Python, JavaScript, C++, or HTML/CSS

Guidelines for language selection:
- Python: Best for data processing, machine learning, automation, scripting, web backends (with frameworks)
- JavaScript: Best for web applications, interactive UIs, Node.js backends, cross-platform mobile apps
- C++: Best for performance-critical applications, system programming, game development, embedded systems
- HTML/CSS: Best for static websites, web interfaces, and frontend styling (usually with JavaScript)

Your output should be a clear decision on which language to use, with a brief justification.
Only respond with one of these exact language choices: "Python", "JavaScript", "C++", or "HTML/CSS".
"""),
    MessagesPlaceholder(variable_name="messages"),
])

# Create the agent runnables
python_coding_agent = python_coding_prompt | llm
javascript_coding_agent = javascript_coding_prompt | llm
cpp_coding_agent = cpp_coding_prompt | llm
html_css_coding_agent = html_css_coding_prompt | llm
language_decider_agent = language_decider_prompt | llm

# Function to decide which language to use
def decide_language(messages: List[Union[AIMessage, HumanMessage, SystemMessage, FunctionMessage]]) -> str:
    """
    Determine which programming language to use based on the task description.
    
    Args:
        messages: The conversation history
        
    Returns:
        A string indicating the chosen language: "Python", "JavaScript", "C++", or "HTML/CSS"
    """
    logger.info("LANGUAGE_DECIDER: Starting language decision process")
    start_time = time.time()
    
    try:
        # Extract the task description from messages
        task_description = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                task_description += message.content + "\n"
        
        # Look for language-specific keywords in the task description
        task_lower = task_description.lower()
        
        # Check for explicit language mentions
        if "use python" in task_lower or "in python" in task_lower:
            logger.info("LANGUAGE_DECIDER: Explicit Python mention detected")
            return "Python"
        elif "use javascript" in task_lower or "in javascript" in task_lower or "use js" in task_lower or "in js" in task_lower:
            logger.info("LANGUAGE_DECIDER: Explicit JavaScript mention detected")
            return "JavaScript"
        elif "use c++" in task_lower or "in c++" in task_lower:
            logger.info("LANGUAGE_DECIDER: Explicit C++ mention detected")
            return "C++"
        elif "use html" in task_lower or "in html" in task_lower or "use css" in task_lower:
            logger.info("LANGUAGE_DECIDER: Explicit HTML/CSS mention detected")
            return "HTML/CSS"
        
        # Check for task-specific indicators
        web_keywords = ["website", "webpage", "web page", "web app", "webapp", "html", "css", "frontend", "front-end", "ui", "user interface"]
        data_keywords = ["data", "analysis", "machine learning", "ml", "ai", "processing", "automation", "script"]
        performance_keywords = ["performance", "speed", "efficient", "memory", "system", "game", "real-time", "realtime"]
        
        web_score = sum(1 for keyword in web_keywords if keyword in task_lower)
        data_score = sum(1 for keyword in data_keywords if keyword in task_lower)
        performance_score = sum(1 for keyword in performance_keywords if keyword in task_lower)
        
        # If we have clear indicators, make a decision
        if web_score > max(data_score, performance_score) and web_score > 1:
            logger.info(f"LANGUAGE_DECIDER: Web task detected (score: {web_score})")
            return "HTML/CSS"
        elif performance_score > max(web_score, data_score) and performance_score > 1:
            logger.info(f"LANGUAGE_DECIDER: Performance-critical task detected (score: {performance_score})")
            return "C++"
        elif data_score > max(web_score, performance_score) and data_score > 1:
            logger.info(f"LANGUAGE_DECIDER: Data processing task detected (score: {data_score})")
            return "Python"
        
        # If no clear indicators, use the LLM to decide
        decider_instruction = HumanMessage(
            content="Based on the task description, which programming language would be most appropriate: Python, JavaScript, C++, or HTML/CSS? Respond with just the language name."
        )
        decider_messages = messages + [decider_instruction]
        
        logger.info("LANGUAGE_DECIDER: Invoking language decider LLM")
        response = language_decider_agent.invoke({"messages": decider_messages})
        logger.info("LANGUAGE_DECIDER: Received response from LLM")
        
        # Extract the language choice from the response
        content = response.content.strip()
        
        # Normalize the language name
        if "python" in content.lower():
            language = "Python"
        elif "javascript" in content.lower() or "js" in content.lower():
            language = "JavaScript"
        elif "c++" in content.lower() or "cpp" in content.lower():
            language = "C++"
        elif "html" in content.lower() or "css" in content.lower() or "web" in content.lower():
            language = "HTML/CSS"
        else:
            # Default to Python if no clear match
            language = "Python"
            logger.warning(f"LANGUAGE_DECIDER: No clear language match in '{content}', defaulting to Python")
        
        elapsed = time.time() - start_time
        logger.info(f"LANGUAGE_DECIDER: Selected language: {language} in {elapsed:.2f} seconds")
        return language
    except Exception as e:
        logger.error(f"Error in language decider: {str(e)}", exc_info=True)
        print(f"Error in language decider: {str(e)}")
        # Default to Python in case of error
        return "Python"

# Function to get the appropriate coding agent based on language
def get_language_coding_agent(language: str):
    """
    Return the appropriate coding agent based on the language.
    
    Args:
        language: The programming language to use
        
    Returns:
        The appropriate language-specific coding agent
    """
    if language == "Python":
        return python_coding_agent
    elif language == "JavaScript":
        return javascript_coding_agent
    elif language == "C++":
        return cpp_coding_agent
    elif language == "HTML/CSS":
        return html_css_coding_agent
    else:
        # Default to Python if language is not recognized
        logger.warning(f"Unrecognized language: {language}, defaulting to Python agent")
        return python_coding_agent

# Function to get appropriate file extension for a language
def get_file_extension(language: str) -> str:
    """
    Return the appropriate file extension for a given language.
    
    Args:
        language: The programming language
        
    Returns:
        The file extension (including the dot)
    """
    if language == "Python":
        return ".py"
    elif language == "JavaScript":
        return ".js"
    elif language == "C++":
        return ".cpp"
    elif language == "HTML/CSS":
        return ".html"
    else:
        return ".txt"

# Main language-specific coding function to be used in the workflow
def language_specific_coding(state: dict, messages: list, plan: str) -> Tuple[str, str]:
    """
    Run the appropriate language-specific coding agent based on the task.
    
    Args:
        state: The current state dictionary
        messages: The conversation history
        plan: The plan created by the planning agent
        
    Returns:
        A tuple containing (code, language)
    """
    logger.info("LANGUAGE_CODING: Starting language-specific coding process")
    start_time = time.time()
    
    try:
        # Determine which language to use
        language = state.get("language", "")
        if not language:
            language = decide_language(messages)
            logger.info(f"LANGUAGE_CODING: Selected language: {language}")
        else:
            logger.info(f"LANGUAGE_CODING: Using previously selected language: {language}")
        
        # Get the appropriate coding agent
        coding_agent = get_language_coding_agent(language)
        
        # Add a specific instruction for the coding agent
        coding_instruction = HumanMessage(
            content=f"""Based on this plan:

{plan}

Please implement the complete code in {language}. Make sure to:
1. Include all necessary imports/dependencies
2. Implement all functions and classes mentioned in the plan
3. Add appropriate comments and documentation
4. If the implementation requires multiple files, clearly indicate each file with a filename header

For multiple files, use this format:
```
// FILENAME: example.js
// Code for example.js goes here
```

```
/* FILENAME: styles.css */
/* Code for styles.css goes here */
```

```python
# FILENAME: app.py
# Code for app.py goes here
```

Make sure each file is properly formatted and includes all necessary code.
"""
        )
        coding_messages = messages + [coding_instruction]
        
        logger.info(f"LANGUAGE_CODING: Invoking {language} coding agent")
        response = coding_agent.invoke({"messages": coding_messages})
        logger.info(f"LANGUAGE_CODING: Received response from {language} agent")
        
        # Extract the code from the response
        content = response.content
        logger.info("LANGUAGE_CODING: Extracting code from response")
        
        # Check if there are multiple files in the response
        file_pattern = r"(?:FILENAME|filename):\s*([^\n]+)"
        filenames = re.findall(file_pattern, content)
        
        if filenames:
            logger.info(f"LANGUAGE_CODING: Detected multiple files: {filenames}")
            
            # Extract code for each file
            files_dict = {}
            
            # Pattern to match file blocks
            file_blocks_pattern = r"(?:FILENAME|filename):\s*([^\n]+)(?:\n|\r\n?)(.*?)(?=(?:\n|\r\n?)(?:FILENAME|filename):|$)"
            file_blocks = re.findall(file_blocks_pattern, content, re.DOTALL)
            
            for filename, file_code in file_blocks:
                # Clean up the filename and code
                clean_filename = filename.strip()
                
                # Remove any markdown code block markers
                clean_code = re.sub(r'^```\w*\n|```$', '', file_code, flags=re.MULTILINE).strip()
                
                # Store in dictionary
                files_dict[clean_filename] = clean_code
                logger.info(f"LANGUAGE_CODING: Extracted code for {clean_filename} ({len(clean_code)} characters)")
            
            # Save all files
            if files_dict:
                save_result = save_multiple_files(files_dict)
                logger.info(f"LANGUAGE_CODING: Multiple files saved: {save_result}")
                
                # For the state, concatenate all code with file headers
                all_code = ""
                for filename, file_code in files_dict.items():
                    all_code += f"# FILENAME: {filename}\n\n{file_code}\n\n"
                
                code = all_code
            else:
                # Fallback to regular code extraction if file parsing failed
                code = extract_code_from_response(content)
                logger.info(f"LANGUAGE_CODING: Extracted single code block ({len(code)} characters)")
        else:
            # Regular code extraction for single file
            code = extract_code_from_response(content)
            logger.info(f"LANGUAGE_CODING: Extracted single code block ({len(code)} characters)")
        
        elapsed = time.time() - start_time
        logger.info(f"LANGUAGE_CODING: Completed in {elapsed:.2f} seconds")
        
        return code, language
    except Exception as e:
        logger.error(f"Error in language-specific coding: {str(e)}", exc_info=True)
        print(f"Error in language-specific coding: {str(e)}")
        # Return empty code in case of error
        return "", "Unknown"
