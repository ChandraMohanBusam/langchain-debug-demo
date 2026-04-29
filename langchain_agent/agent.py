"""
agent.py
AI Deployment Agent - Main Entry Point

This module demonstrates three debugging levels for a LangChain ReAct agent:

LEVEL 1 - verbose=True and set_debug(True)
    Fast, terminal-only, overwhelming but useful for first look.

LEVEL 2 - FileCallbackHandler
    Persistent file logging with zero custom code.

LEVEL 3 - Custom Callbacks (DeploymentAuditHandler, TokenUsageHandler,
           LoopDetectionHandler, SlackAlertHandler)
    Surgical precision. Production-grade. The right tool for the job.

Run:
    python agent.py --mode level1   # verbose + debug mode
    python agent.py --mode level2   # FileCallbackHandler
    python agent.py --mode level3   # full custom callbacks (default)
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.file import FileCallbackHandler
from langchain.globals import set_debug, set_verbose
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from tools import DownloadBuild, DeployOnServer, RestartServices, TransferToServer, reset_server_state

load_dotenv()

# ---------------------------------------------------------------------------
# Agent Prompt
# ReAct format: the agent reasons step by step and selects tools accordingly.
# ---------------------------------------------------------------------------
DEPLOYMENT_PROMPT = PromptTemplate.from_template(
    """You are an AI Deployment Agent responsible for deploying software builds to production servers.

You must follow these steps IN ORDER:
1. Download the build using DownloadBuild
2. Transfer the build to the server using TransferToServer
3. ONLY proceed to DeployOnServer if TransferToServer confirms checksum verified
4. After successful deployment, restart services using RestartServices

If TransferToServer reports a checksum failure, retry it. Do not proceed to DeployOnServer
until you see "Checksum verified" in the tool output.

You have access to the following tools:
{tools}

Use the following format:

Question: the deployment task you must complete
Thought: your reasoning about what to do next
Action: the tool to use, must be one of [{tool_names}]
Action Input: the input to the tool
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have completed all deployment steps
Final Answer: summary of the completed deployment

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

DEPLOYMENT_TOOLS = [DownloadBuild, TransferToServer, DeployOnServer, RestartServices]


def build_llm(callbacks=None):
    """Creates the ChatOpenAI LLM instance with optional callbacks."""
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        callbacks=callbacks or [],
        # Note: Do NOT use streaming=True if you need token usage tracking.
        # Token counts are not returned in streaming mode.
    )


def run_level1():
    """
    LEVEL 1: verbose=True and set_debug(True)

    Best for: Quick first look when you have no idea what is happening.
    Drawback: Floods the terminal. No persistent logs. Hard to grep.
    """
    print("\n" + "="*60)
    print("DEBUGGING LEVEL 1: verbose=True + set_debug(True)")
    print("="*60)
    print("WARNING: This will print a LOT of output to the terminal.\n")

    # Enable global debug mode - prints every component's input/output
    set_debug(True)
    set_verbose(True)

    llm = build_llm()
    agent = create_react_agent(llm, DEPLOYMENT_TOOLS, DEPLOYMENT_PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=DEPLOYMENT_TOOLS,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )

    result = executor.invoke({
        "input": "Deploy build version 3.2.1 to server 192.168.1.100"
    })
    print(f"\nFinal Result: {result['output']}")

    # Turn off debug after the run
    set_debug(False)
    set_verbose(False)


def run_level2():
    """
    LEVEL 2: FileCallbackHandler

    Best for: When you want persistent logs without writing any custom code.
    LangChain's built-in file logger writes formatted output to a text file.
    Drawback: Fixed format, no filtering, no alerting.
    """
    print("\n" + "="*60)
    print("DEBUGGING LEVEL 2: FileCallbackHandler")
    print("="*60)
    print("Logs will be written to: logs/langchain_file_callback.log\n")

    os.makedirs("logs", exist_ok=True)
    file_handler = FileCallbackHandler("logs/langchain_file_callback.log")

    llm = build_llm(callbacks=[file_handler])
    agent = create_react_agent(llm, DEPLOYMENT_TOOLS, DEPLOYMENT_PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=DEPLOYMENT_TOOLS,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
        callbacks=[file_handler],
    )

    result = executor.invoke({
        "input": "Deploy build version 3.2.1 to server 192.168.1.100"
    })
    print(f"\nFinal Result: {result['output']}")
    print("Check logs/langchain_file_callback.log for the full trace.")


def run_level3():
    """
    LEVEL 3: Config-driven Custom Callback Stack

    All handlers and alert channels are controlled by config.json.
    No code changes needed to toggle logging or switch Slack vs Teams.

    Edit config.json to:
    - Enable/disable individual log files
    - Adjust loop detection threshold
    - Switch alert provider between "slack", "teams", or "none"
    - Suppress specific alert types (llm_error, tool_error, etc.)

    ConfigLoader reads config.json at startup and builds the exact
    handler list dynamically.
    """
    print("\n" + "="*60)
    print("DEBUGGING LEVEL 3: Config-driven Custom Callback Stack")
    print("="*60)
    print("Handler configuration is read from: config.json\n")

    from config_loader import ConfigLoader

    # One call builds the entire handler stack based on config.json
    all_handlers = ConfigLoader.build_handlers()

    llm = build_llm(callbacks=all_handlers)
    agent = create_react_agent(llm, DEPLOYMENT_TOOLS, DEPLOYMENT_PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=DEPLOYMENT_TOOLS,
        verbose=False,         # Keep terminal clean - logs go to files
        max_iterations=12,     # Safety ceiling to prevent runaway loops
        handle_parsing_errors=True,
        callbacks=all_handlers,
    )

    print("Starting deployment agent run...\n")
    try:
        result = executor.invoke({
            "input": "Deploy build version 3.2.1 to server 192.168.1.100"
        })
        print(f"\nFinal Result: {result['output']}")
    except Exception as e:
        print(f"\nAgent failed with error: {e}")
        print("Check logs/deployment_audit.log for the full trace.")

    print("\nRun complete. Check the logs/ directory for detailed output.")


def main():
    parser = argparse.ArgumentParser(
        description="AI Deployment Agent - LangChain Debugging Demo"
    )
    parser.add_argument(
        "--mode",
        choices=["level1", "level2", "level3"],
        default="level3",
        help=(
            "level1: verbose+debug | "
            "level2: FileCallbackHandler | "
            "level3: full custom callbacks (default)"
        ),
    )
    args = parser.parse_args()

    reset_server_state()

    if args.mode == "level1":
        run_level1()
    elif args.mode == "level2":
        run_level2()
    else:
        run_level3()


if __name__ == "__main__":
    main()
