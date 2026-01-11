#!/usr/bin/env python3
"""Show the final response for Kashmir pickup query."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Prevent torch loading issues
os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")

# Configure LangSmith BEFORE importing graph modules (IMPORTANT for tracing!)
from app.settings import Settings
settings = Settings()

if settings.langsmith_tracing:
    langsmith_key = settings.effective_langsmith_api_key()
    if langsmith_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_PROJECT"] = settings.effective_langsmith_project()
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        # Handle workspace ID for org-scoped API keys (optional)
        workspace_id = (
            os.getenv("LANGSMITH_WORKSPACE_ID") or 
            settings.langsmith_workspace_id or
            os.getenv("LANGCHAIN_WORKSPACE_ID")
        )
        if workspace_id:
            os.environ["LANGSMITH_WORKSPACE_ID"] = workspace_id
            os.environ["LANGCHAIN_WORKSPACE_ID"] = workspace_id
    elif os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = settings.effective_langsmith_project()
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Suppress LangSmith warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# NOW import build_graph (after env vars are set)
from graph.build_graph import build_graph
from graph.state import InputPayload, Questions

def main():
    """Run Kashmir pickup query and show final response."""
    query = "Is pickup included in the Kashmir trip, or do I need to reach Srinagar on my own?"
    
    print("\n" + "=" * 70)
    print("KASHMIR PICKUP QUERY - FINAL RESPONSE")
    print("=" * 70)
    print(f"\nQuery: {query}\n")
    
    try:
        # Build graph
        graph = build_graph()
        
        # Create initial state
        initial_state = {
            "input": InputPayload(raw_text=query),
            "questions": Questions()
        }
        
        # Run workflow
        print("Running workflow...\n")
        final_state = graph.invoke(initial_state)
        
        # Get merged output
        merged_output = final_state.get("merged_output", {})
        final_text = merged_output.get("final_text", "No output generated")
        
        # Get trip context
        answerable_processing = final_state.get("answerable_processing", {})
        trip_context = answerable_processing.get("trip_context", {})
        trip_id = trip_context.get("trip_id", "Not resolved")
        confidence = trip_context.get("confidence", "N/A")
        
        # Display results
        print("=" * 70)
        print("FINAL RESPONSE:")
        print("=" * 70)
        print(final_text)
        print("=" * 70)
        
        print(f"\nTrip Resolution Details:")
        print(f"  Trip ID: {trip_id}")
        print(f"  Confidence: {confidence}")
        print("=" * 70)
        
        return final_text
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
