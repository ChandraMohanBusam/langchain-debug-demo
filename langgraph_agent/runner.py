"""
runner.py
Document Review Agent - Main Entry Point

Demonstrates LangGraph-specific debugging tools that do NOT exist in LangChain:

MODE 1 - Standard run (no debugging)
    Just run the graph and see the final output.

MODE 2 - stream_mode="debug"
    Real-time granular event stream. Shows every task trigger and state update
    as the graph executes. LangGraph's equivalent of verbose=True, but structured.

MODE 3 - Checkpointer + get_state()
    After the run, inspect the final state snapshot with get_state().
    See exactly what the State object looks like at the end of execution.

MODE 4 - State History + Time Travel
    List every checkpoint saved during the run with get_state_history().
    Rewind to any past checkpoint and replay from that point.
    This is how you debug a loop: find the last clean state before things went wrong.

Run:
    python runner.py --mode standard
    python runner.py --mode stream_debug
    python runner.py --mode inspect_state
    python runner.py --mode time_travel    (default)
"""

import argparse
import os
from datetime import datetime

from dotenv import load_dotenv

from graph import build_graph, DocumentReviewState
from teams_alert import send_teams_alert

load_dotenv()

# ---------------------------------------------------------------------------
# Sample document that triggers the loop scenario.
# The content is vague enough that the reviewer keeps requesting revisions.
# ---------------------------------------------------------------------------
SAMPLE_DOCUMENT = {
    "document_title": "Q3 Vendor Agreement Draft",
    "document_content": (
        "This agreement is between two parties. The terms are to be discussed. "
        "Payment will be made at some point. Services will be rendered as needed. "
        "Both parties agree to things generally."
    ),
    "document_type": "",
    "review_notes": "",
    "decision": "",
    "revision_count": 0,
    "final_outcome": "",
    "messages": [],
}

THREAD_CONFIG = {"configurable": {"thread_id": "review_session_001"}}


def run_standard():
    """
    MODE 1: Standard run, no debugging.
    Just invoke the graph and print the final output.
    """
    print("\n" + "="*60)
    print("MODE 1: Standard Run (No Debugging)")
    print("="*60)

    graph = build_graph(use_checkpointer=False)
    result = graph.invoke(SAMPLE_DOCUMENT)
    print(f"\nFinal Outcome: {result.get('final_outcome', 'No outcome set')}")
    print(f"Revision Count: {result.get('revision_count', 0)}")


def run_stream_debug():
    """
    MODE 2: stream_mode="debug"

    LangGraph's built-in debug stream emits a structured event for every:
    - task execution start/end
    - state update (what changed after each node)
    - checkpoint saved

    This is far more useful than LangChain's verbose=True because the output
    is structured (dicts, not raw text) and shows state diffs, not just text.
    """
    print("\n" + "="*60)
    print("MODE 2: stream_mode='debug' - Real-time State Events")
    print("="*60)
    print("Each event shows: type, node name, and state changes.\n")

    graph = build_graph(use_checkpointer=True)
    event_count = 0

    for event in graph.stream(
        SAMPLE_DOCUMENT,
        config=THREAD_CONFIG,
        stream_mode="debug",
    ):
        event_count += 1
        event_type = event.get("type", "unknown")
        node_name = event.get("step", event.get("payload", {}).get("name", ""))

        if event_type == "task":
            payload = event.get("payload", {})
            print(
                f"[Event #{event_count}] TASK | "
                f"node={payload.get('name', 'unknown')} | "
                f"id={payload.get('id', '')[:8]}..."
            )
        elif event_type == "task_result":
            payload = event.get("payload", {})
            result_keys = list(payload.get("result", {}).keys()) if payload.get("result") else []
            print(
                f"[Event #{event_count}] TASK RESULT | "
                f"node={payload.get('name', 'unknown')} | "
                f"state_keys_updated={result_keys}"
            )
        elif event_type == "checkpoint":
            print(
                f"[Event #{event_count}] CHECKPOINT SAVED | "
                f"step={event.get('step', 'unknown')}"
            )
        else:
            print(f"[Event #{event_count}] {event_type.upper()}")

    print(f"\nTotal events received: {event_count}")


def run_inspect_state():
    """
    MODE 3: Checkpointer + get_state()

    After the run, call get_state() to see the full State snapshot.
    This answers: "What does the agent know right now?"

    Unlike LangChain where you have to parse log files to understand state,
    LangGraph gives you a clean Python dict you can inspect programmatically.
    """
    print("\n" + "="*60)
    print("MODE 3: Checkpointer + get_state() Inspection")
    print("="*60)

    graph = build_graph(use_checkpointer=True)

    # Run the graph
    print("Running graph...")
    result = graph.invoke(SAMPLE_DOCUMENT, config=THREAD_CONFIG)

    # Inspect final state
    print("\n--- FINAL STATE SNAPSHOT ---")
    final_state = graph.get_state(THREAD_CONFIG)

    print(f"Values in state:")
    for key, value in final_state.values.items():
        if key == "messages":
            print(f"  messages: [{len(value)} message(s)]")
        else:
            display_value = str(value)[:120] if value else "(empty)"
            print(f"  {key}: {display_value}")

    print(f"\nNext node(s) to run: {final_state.next}")
    print(f"Config: {final_state.config}")

    # This is the key insight: if final_state.next is not empty,
    # the graph is paused mid-run (e.g., waiting for human input or interrupted).
    if final_state.next:
        print(
            f"\nWARNING: Graph is paused. "
            f"Next node is '{final_state.next}'. "
            f"The run did not complete normally."
        )


def run_time_travel():
    """
    MODE 4: State History + Time Travel

    This is the most powerful LangGraph-specific debugging tool.

    get_state_history() returns a list of every checkpoint saved during the run,
    from newest to oldest. Each checkpoint has the full State at that point in time.

    Time Travel = pick any checkpoint and re-invoke the graph from that point.
    This lets you:
    1. Find the exact checkpoint where the loop started
    2. Fix your code
    3. Resume from the last clean state, not from the beginning

    This capability does NOT exist in standard LangChain chains.
    """
    print("\n" + "="*60)
    print("MODE 4: State History + Time Travel")
    print("="*60)

    graph = build_graph(use_checkpointer=True)

    # Run the graph (it may loop through revisions)
    print("Running graph (watch for revision loops)...\n")
    try:
        result = graph.invoke(
            SAMPLE_DOCUMENT,
            config=THREAD_CONFIG,
        )
    except Exception as e:
        print(f"Graph run ended with: {e}")

    # List all checkpoints (most recent first)
    print("\n--- STATE HISTORY (most recent first) ---")
    history = list(graph.get_state_history(THREAD_CONFIG))

    for i, checkpoint in enumerate(history):
        revision_count = checkpoint.values.get("revision_count", 0)
        decision = checkpoint.values.get("decision", "")
        next_nodes = checkpoint.next

        print(
            f"Checkpoint #{i} | "
            f"next={next_nodes} | "
            f"revision_count={revision_count} | "
            f"decision='{decision}' | "
            f"id={str(checkpoint.config.get('configurable', {}).get('checkpoint_id', ''))[:12]}..."
        )

    # Time Travel Demo: rewind to the first revision checkpoint
    # Find the checkpoint just before the first revision
    print("\n--- TIME TRAVEL DEMO ---")
    print("Finding the checkpoint just before the first revision...")

    pre_revision_checkpoint = None
    for checkpoint in reversed(history):
        if checkpoint.values.get("revision_count", 0) == 0 and checkpoint.next:
            pre_revision_checkpoint = checkpoint
            break

    if pre_revision_checkpoint:
        print(
            f"Found pre-revision checkpoint. "
            f"Next node was: {pre_revision_checkpoint.next}"
        )
        print(
            "In a real scenario, you would fix your code here, then replay "
            "from this checkpoint using:\n"
            "    graph.invoke(None, config=pre_revision_checkpoint.config)\n"
            "This resumes from the saved state without restarting from scratch."
        )

        # Show what the state looked like at that checkpoint
        print("\nState at pre-revision checkpoint:")
        for key, value in pre_revision_checkpoint.values.items():
            if key != "messages":
                display = str(value)[:100] if value else "(empty)"
                print(f"  {key}: {display}")
    else:
        print("No pre-revision checkpoint found (document was approved on first review).")

    print(f"\nFinal outcome: {result.get('final_outcome', 'N/A')}")
    print(f"Total revisions: {result.get('revision_count', 0)}")


def main():
    parser = argparse.ArgumentParser(
        description="Document Review Agent - LangGraph Debugging Demo"
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "stream_debug", "inspect_state", "time_travel"],
        default="time_travel",
        help=(
            "standard: basic run | "
            "stream_debug: real-time event stream | "
            "inspect_state: get_state() after run | "
            "time_travel: full history + replay demo (default)"
        ),
    )
    args = parser.parse_args()

    if args.mode == "standard":
        run_standard()
    elif args.mode == "stream_debug":
        run_stream_debug()
    elif args.mode == "inspect_state":
        run_inspect_state()
    else:
        run_time_travel()


if __name__ == "__main__":
    main()
