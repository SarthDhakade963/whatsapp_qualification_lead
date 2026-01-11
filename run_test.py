#!/usr/bin/env python3
"""Test script to run the conversational workflow graph with LangSmith tracing."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Prevent torch loading issues on Windows (set before any imports)
os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")

# Configure LangSmith before importing (if using env vars)
from app.settings import Settings
settings = Settings()

if settings.langsmith_tracing:
    langsmith_key = settings.effective_langsmith_api_key()
    if langsmith_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_PROJECT"] = settings.effective_langsmith_project()
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        # Handle workspace ID for org-scoped API keys (optional - set if available)
        workspace_id = (
            os.getenv("LANGSMITH_WORKSPACE_ID") or 
            settings.langsmith_workspace_id or
            os.getenv("LANGCHAIN_WORKSPACE_ID")
        )
        if workspace_id:
            os.environ["LANGSMITH_WORKSPACE_ID"] = workspace_id
            os.environ["LANGCHAIN_WORKSPACE_ID"] = workspace_id

from graph.build_graph import build_graph
from graph.state import ConversationWorkflowState, InputPayload, Questions


def main():
    """Run the test scenario."""
    
    # Build the graph (tracing configured in build_graph)
    graph = build_graph()
    
    # Create initial state as dict (TypedDict requires dict input)
    initial_state = {
        "input": InputPayload(raw_text="Is pickup included and what about refunds?"),
        "questions": Questions()
    }
    
    # Run the graph
    print("Running workflow...")
    input_text = initial_state["input"].raw_text if hasattr(initial_state["input"], "raw_text") else initial_state["input"].get("raw_text", "")
    print(f"Input: {input_text}")
    print("-" * 50)
    
    final_state = graph.invoke(initial_state)
    
    # Print output (LangGraph returns dict for TypedDict state)
    merged_output = final_state.get("merged_output")
    if merged_output:
        final_text = merged_output.get("final_text") if isinstance(merged_output, dict) else merged_output.final_text if hasattr(merged_output, "final_text") else ""
        print("\nOutput:")
        print(final_text)
    else:
        print("\nNo output generated.")
    
    return final_state


if __name__ == "__main__":
    main()

