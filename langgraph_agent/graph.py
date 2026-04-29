"""
graph.py
Document Review Agent - LangGraph State Machine

This agent reviews incoming documents through a multi-node pipeline:

    START -> classify -> review -> [approve | reject | revise]
                                        |
                               revise --+  (loop scenario)

The loop scenario: if the reviewer keeps requesting revisions,
the agent can get stuck cycling between 'review' and 'revise'.

This module demonstrates LangGraph-specific debugging:
- Explicit State object inspection at each node
- Checkpointers (MemorySaver) for state snapshots
- stream_mode="debug" for real-time granular event logging
- get_state() for post-run state inspection
- Time Travel: replaying from a saved checkpoint
"""

import os
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from teams_alert import send_teams_alert

# ---------------------------------------------------------------------------
# State Definition
# The State object is the "single source of truth" in LangGraph.
# Unlike LangChain chains where data flows implicitly forward,
# LangGraph makes state explicit and inspectable at every node.
# ---------------------------------------------------------------------------
class DocumentReviewState(TypedDict):
    document_title: str
    document_content: str
    document_type: str          # Set by classify node
    review_notes: str           # Set by review node
    decision: str               # "approve", "reject", "revise"
    revision_count: int         # Tracks how many times we have revised
    final_outcome: str          # Set when the process completes
    messages: Annotated[list, add_messages]  # Full message history


# Loop guard: if revision_count exceeds this, force a rejection
MAX_REVISIONS = 3

# ---------------------------------------------------------------------------
# LLM Setup
# ---------------------------------------------------------------------------
def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------------------------
# Node Definitions
# Each node receives the full State and returns a partial update.
# LangGraph merges the returned dict back into State automatically.
# ---------------------------------------------------------------------------

def classify_node(state: DocumentReviewState) -> dict:
    """
    Node 1: Classify the document type.
    Determines: contract, invoice, report, policy, or unknown.
    """
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "You are a document classifier. Classify the document into one of: "
            "contract, invoice, report, policy, unknown. "
            "Respond with ONLY the document type, nothing else."
        )),
        HumanMessage(content=(
            f"Document Title: {state['document_title']}\n"
            f"Content Preview: {state['document_content'][:500]}"
        )),
    ])

    doc_type = response.content.strip().lower()
    print(f"[classify_node] Document type: {doc_type}")

    return {
        "document_type": doc_type,
        "messages": [response],
    }


def review_node(state: DocumentReviewState) -> dict:
    """
    Node 2: Review the document and decide action.
    Decision: approve, reject, or revise.

    The loop scenario: the LLM keeps requesting revisions because
    the document content is intentionally vague. This causes
    the graph to cycle between review_node and revise_node.
    """
    llm = get_llm()
    revision_count = state.get("revision_count", 0)

    # If we have already revised too many times, force approval to break loop
    # This is the guard we added AFTER discovering the loop via checkpointers
    if revision_count >= MAX_REVISIONS:
        print(
            f"[review_node] Max revisions ({MAX_REVISIONS}) reached. "
            f"Forcing approval to break loop."
        )
        return {
            "decision": "approve",
            "review_notes": (
                f"Approved after {revision_count} revisions. "
                f"Loop guard triggered at MAX_REVISIONS={MAX_REVISIONS}."
            ),
        }

    response = llm.invoke([
        SystemMessage(content=(
            f"You are a strict document reviewer. The document type is: {state['document_type']}.\n"
            f"Review the document and decide: approve, reject, or revise.\n"
            f"This document has been revised {revision_count} time(s) already.\n"
            f"If revision_count > 1, be more lenient and lean toward approving.\n"
            "Respond in this exact format:\n"
            "DECISION: [approve|reject|revise]\n"
            "NOTES: [your review notes]"
        )),
        HumanMessage(content=(
            f"Title: {state['document_title']}\n"
            f"Content: {state['document_content']}\n"
            f"Previous notes: {state.get('review_notes', 'None')}"
        )),
    ])

    content = response.content.strip()
    lines = content.split("\n")

    decision = "revise"
    notes = content

    for line in lines:
        if line.startswith("DECISION:"):
            decision = line.replace("DECISION:", "").strip().lower()
        elif line.startswith("NOTES:"):
            notes = line.replace("NOTES:", "").strip()

    print(f"[review_node] Decision: {decision} | Revision #{revision_count}")

    return {
        "decision": decision,
        "review_notes": notes,
        "messages": [response],
    }


def revise_node(state: DocumentReviewState) -> dict:
    """
    Node 3: Apply revision suggestions to the document.
    Increments revision_count each time. This is how we detect the loop.
    """
    llm = get_llm()
    revision_count = state.get("revision_count", 0) + 1

    print(f"[revise_node] Applying revision #{revision_count}")

    # Send Teams alert if revision count is high (potential loop detected)
    if revision_count >= 2:
        send_teams_alert(
            title="Revision Loop Warning - Document Review Agent",
            message=(
                f"Document '{state['document_title']}' is on revision #{revision_count}. "
                f"Agent may be stuck in a review/revise loop."
            ),
            severity="warning",
        )

    response = llm.invoke([
        SystemMessage(content=(
            "You are a document editor. Apply the reviewer's notes to improve the document. "
            "Return only the revised document content, nothing else."
        )),
        HumanMessage(content=(
            f"Original content: {state['document_content']}\n"
            f"Reviewer notes: {state['review_notes']}"
        )),
    ])

    return {
        "document_content": response.content.strip(),
        "revision_count": revision_count,
        "messages": [response],
    }


def approve_node(state: DocumentReviewState) -> dict:
    """Node 4a: Document approved. Sets final outcome."""
    print(f"[approve_node] Document approved after {state.get('revision_count', 0)} revision(s).")
    return {
        "final_outcome": (
            f"APPROVED: '{state['document_title']}' approved after "
            f"{state.get('revision_count', 0)} revision(s). "
            f"Notes: {state.get('review_notes', '')}"
        )
    }


def reject_node(state: DocumentReviewState) -> dict:
    """Node 4b: Document rejected. Sends Teams alert and sets final outcome."""
    print(f"[reject_node] Document rejected.")

    send_teams_alert(
        title="Document Rejected - Review Agent",
        message=(
            f"Document '{state['document_title']}' has been rejected.\n"
            f"Reason: {state.get('review_notes', 'No notes provided')}"
        ),
        severity="error",
    )

    return {
        "final_outcome": (
            f"REJECTED: '{state['document_title']}' rejected. "
            f"Reason: {state.get('review_notes', '')}"
        )
    }


# ---------------------------------------------------------------------------
# Routing Function
# Determines next node based on the decision in State.
# This is where loops can form: if decision is always "revise",
# the graph cycles between review_node and revise_node.
# ---------------------------------------------------------------------------
def route_decision(
    state: DocumentReviewState,
) -> Literal["revise_node", "approve_node", "reject_node"]:
    """Routes to the next node based on the reviewer's decision."""
    decision = state.get("decision", "revise")
    if decision == "approve":
        return "approve_node"
    elif decision == "reject":
        return "reject_node"
    else:
        return "revise_node"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------
def build_graph(use_checkpointer: bool = True):
    """
    Builds and compiles the Document Review graph.

    Args:
        use_checkpointer: If True, attaches a MemorySaver checkpointer.
            This enables:
            - get_state() for state inspection
            - get_state_history() for time travel
            - stream_mode="debug" for granular events
    """
    builder = StateGraph(DocumentReviewState)

    # Add nodes
    builder.add_node("classify_node", classify_node)
    builder.add_node("review_node", review_node)
    builder.add_node("revise_node", revise_node)
    builder.add_node("approve_node", approve_node)
    builder.add_node("reject_node", reject_node)

    # Add edges
    builder.add_edge(START, "classify_node")
    builder.add_edge("classify_node", "review_node")

    # Conditional routing from review_node
    builder.add_conditional_edges(
        "review_node",
        route_decision,
        {
            "revise_node": "revise_node",
            "approve_node": "approve_node",
            "reject_node": "reject_node",
        },
    )

    # After revision, go back to review (this is where loops happen)
    builder.add_edge("revise_node", "review_node")

    # Terminal nodes
    builder.add_edge("approve_node", END)
    builder.add_edge("reject_node", END)

    if use_checkpointer:
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)
    else:
        return builder.compile()
