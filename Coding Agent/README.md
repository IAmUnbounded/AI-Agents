# Multi-Agent Code Writing System

This project implements a multi-agent system for collaborative code writing using LangGraph and the Gemini API.

## System Architecture

The system consists of five specialized agents that work together to write code:

1. **Orchestrator Agent**: Coordinates the workflow, understands requirements, and decides which agent should work next.
2. **Planning Agent**: Creates detailed plans and architecture designs based on requirements.
3. **Coding Agent**: Implements the actual code following the plan created by the Planning Agent.
4. **Testing Agent**: Runs the code and identifies any errors or issues that need to be fixed.
5. **Checking Agent**: Reviews the code for bugs, improvements, and adherence to requirements.

## Features

- **Multi-file Support**: The system can generate and manage multiple interconnected files for complex projects.
- **Automatic Testing**: Code is automatically tested to identify runtime errors and issues.
- **Intelligent Workflow Management**: The orchestrator intelligently routes between agents based on the current state.
- **Error Handling**: Robust error handling for API failures and execution issues.
- **Support for Multiple Languages**: Can generate Python, HTML, CSS, and JavaScript code.
- **Iteration Limiting**: Prevents infinite loops with configurable iteration limits.

## Files

- `code_writing_agents.py`: Core implementation of the multi-agent system
- `code_agents_example.py`: Example usage with a FastAPI todo application

## Requirements

- Python 3.8+
- LangGraph
- LangChain
- Google Generative AI Python SDK
- Google API Key (set as environment variable `GOOGLE_API_KEY`)

## Usage

1. Set up your Google API key:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

2. Run the basic example:
```bash
python code_writing_agents.py
```

3. For custom tasks, modify the `initial_task` variable in the main section of `code_writing_agents.py`.

## How It Works

The agents communicate through a state graph managed by LangGraph. The workflow follows these steps:

1. The user provides a coding task
2. The Orchestrator analyzes the task and delegates to the Planning Agent
3. The Planning Agent creates a detailed implementation plan
4. The Orchestrator reviews the plan and delegates to the Coding Agent
5. The Coding Agent implements the code based on the plan
6. The Orchestrator delegates to the Testing Agent
7. The Testing Agent runs the code and identifies any issues
8. If issues are found, the Orchestrator routes back to Planning or Coding
9. Once code passes testing, the Orchestrator delegates to the Checking Agent
10. The Checking Agent reviews the code and provides feedback
11. The Orchestrator decides if further iterations are needed or if the task is complete

## Multi-file Support

The system can generate and manage multiple files for complex projects:

1. The Coding Agent can create multiple files with clear file headers
2. The Testing Agent can identify and test the appropriate main file
3. All files are saved with the correct extensions and content

## Extending the System

You can extend this system by:
- Adding more specialized agents (e.g., Documentation Agent, Security Agent)
- Enhancing the tools available to agents
- Implementing more complex workflows
- Connecting to external services or APIs
