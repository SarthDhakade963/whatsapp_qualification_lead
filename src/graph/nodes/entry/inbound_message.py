from typing import TypedDict
from graph.state import ConversationWorkflowState, InputPayload


def inbound_message(state: ConversationWorkflowState) -> ConversationWorkflowState:
    """
    Entry node: Receives raw input and initializes state.
    Only modifies: input (but only if needed)
    """
    # Input should already be set correctly, just pass through
    # Don't modify if already correct to avoid LangGraph conflicts
    return state

