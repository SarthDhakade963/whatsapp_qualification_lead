#!/usr/bin/env python3
"""Quick test to show the final response for Kashmir pickup query."""

import sys
import os

# Add src to path - use absolute path to handle running from different directories
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
_src_dir = os.path.join(_project_root, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

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

# NOW import build_graph (after env vars are set)
from graph.build_graph import build_graph
from graph.state import InputPayload, Questions
from state.store import StateStore
from state.memory import ConversationMemory

def test_conversation(messages: list[str]):
    """
    Test a WhatsApp conversation with multiple messages.
    Simulates real conversation state loading and saving.
    Each message loads previous conversation history and saves updated state.
    
    Args:
        messages: List of user messages to process sequentially
    """
    print("=" * 80)
    print("WHATSAPP CONVERSATION TEST")
    print("=" * 80)
    print(f"\nProcessing {len(messages)} message(s)...\n")
    
    # Initialize storage (in production, this would be Redis/DB)
    store = StateStore()
    memory = ConversationMemory(store)
    
    # Simulate a session (in production, this comes from WhatsApp phone number)
    session_id = "test_session_123"
    
    # Build graph once (can be reused for all messages)
    graph = build_graph()
    
    # Store conversation summary for display
    conversation_summary = []
    
    # Process each message
    for i, message in enumerate(messages, 1):
        print("\n" + "=" * 80)
        print(f"MESSAGE {i}/{len(messages)}")
        print("=" * 80)
        print(f"\n👤 USER: {message}\n")
        
        # ============================================================
        # STEP 1: LOAD CONVERSATION STATE (Before Processing)
        # ============================================================
        print("📥 Loading conversation state...")
        loaded_history = memory.get_history(session_id)
        recent_history = memory.get_recent_history(session_id, max_messages=6, max_gap_hours=36.0)
        conversation_state = memory.get_or_create_conversation_state(session_id)
        print(f"   Loaded {len(loaded_history)} total message(s) from storage")
        print(f"   Recent messages (within 36h): {len(recent_history)} message(s)")
        print(f"   Conversation State v{conversation_state.get('version', 0)}:")
        print(f"     Focus: {conversation_state.get('focus', {}).get('primary_topic', 'None')} (confidence: {conversation_state.get('focus', {}).get('confidence', 0):.2f})")
        print(f"     Intent: {conversation_state.get('intent_level', 'browsing')} | Risk: {conversation_state.get('risk_level', 'low')}")
        if recent_history:
            print(f"   Recent messages: {[msg.get('content', '')[:30] + '...' for msg in recent_history[-2:] if msg.get('role') == 'user']}")
        
        # ============================================================
        # STEP 2: INITIALIZE STATE WITH LOADED HISTORY AND STATE
        # ============================================================
        initial_state = {
            "input": InputPayload(raw_text=message),
            "questions": Questions(),
            "conversation_history": recent_history if recent_history else None,
            "conversation_state": conversation_state
        }
        
        # ============================================================
        # STEP 3: RUN GRAPH WORKFLOW
        # ============================================================
        print("🔄 Running graph workflow...")
        final_state = graph.invoke(initial_state)
        
        # Extract response
        merged_output = final_state.get("merged_output", {})
        final_text = merged_output.get("final_text", "No output generated")
        
        # Extract trip context if available
        answerable_processing = final_state.get("answerable_processing", {})
        trip_context = answerable_processing.get("trip_context", {}) if answerable_processing else {}
        trip_id = trip_context.get("trip_id", "Not resolved") if trip_context else "Not resolved"
        confidence = trip_context.get("confidence", "N/A") if trip_context else "N/A"
        
        # Extract interaction state
        interaction_state = final_state.get("interaction_state", {})
        decision_stage = interaction_state.get("decision_stage", "N/A") if interaction_state else "N/A"
        escalation_flag = interaction_state.get("escalation_flag", False) if interaction_state else False
        
        # Display response
        print("\n" + "-" * 80)
        print("🤖 BOT RESPONSE:")
        print("-" * 80)
        print(final_text)
        print("-" * 80)
        
        # Display metadata
        print(f"\n📊 Metadata:")
        print(f"  Trip ID: {trip_id}")
        print(f"  Confidence: {confidence}")
        print(f"  Decision Stage: {decision_stage}")
        print(f"  Escalation Flag: {escalation_flag}")
        
        # ============================================================
        # STEP 4: SAVE UPDATED CONVERSATION STATE (After Processing)
        # ============================================================
        print("\n💾 Saving conversation state...")
        
        # Add user message to history
        memory.add_message(session_id, {
            "role": "user",
            "content": message
        })
        
        # Add bot response to history
        memory.add_message(session_id, {
            "role": "assistant",
            "content": final_text
        })
        
        # Update authoritative conversation state
        updated_state = memory.update_conversation_state(
            session_id,
            trip_context=trip_context if trip_id != "Not resolved" else None,
            interaction_state=interaction_state if interaction_state else None
        )
        
        updated_history = memory.get_history(session_id)
        print(f"   Saved 2 new messages (user + assistant)")
        print(f"   Total messages in history: {len(updated_history)}")
        print(f"   Updated conversation state v{updated_state.get('version', 0)}:")
        print(f"     Focus: {updated_state.get('focus', {}).get('primary_topic', 'None')} (confidence: {updated_state.get('focus', {}).get('confidence', 0):.2f})")
        print(f"     Intent: {updated_state.get('intent_level', 'browsing')} | Risk: {updated_state.get('risk_level', 'low')} | Momentum: {updated_state.get('momentum_state', 'building')}")
        if updated_state.get('topic_decay'):
            print(f"     Topic Decay: {updated_state.get('topic_decay')}")
        
        # Store for summary
        conversation_summary.append({
            "message_num": i,
            "user_message": message,
            "bot_response": final_text,
            "trip_id": trip_id,
            "confidence": confidence,
            "decision_stage": decision_stage,
            "escalation_flag": escalation_flag
        })
        
        print("=" * 80)
    
    # Summary
    print("\n" + "=" * 80)
    print("CONVERSATION SUMMARY")
    print("=" * 80)
    final_history = memory.get_history(session_id)
    final_state = memory.get_or_create_conversation_state(session_id)
    print(f"\nFinal conversation history in storage: {len(final_history)} messages")
    for entry in conversation_summary:
        print(f"\nMessage {entry['message_num']}:")
        print(f"  User: {entry['user_message'][:60]}...")
        print(f"  Bot: {entry['bot_response'][:60]}...")
        print(f"  Trip ID: {entry['trip_id']} | Confidence: {entry['confidence']}")
    
    print("\n" + "=" * 80)
    print("FINAL CONVERSATION STATE (Authoritative)")
    print("=" * 80)
    import json
    print(json.dumps(final_state, indent=2))
    print("=" * 80)
    
    return conversation_summary


def main():
    """Run Kashmir pickup query and show final response."""
    query = "“What kind of weather should I expect during the trip, and will there definitely be snowfall?”"
    
    print("=" * 70)
    print("KASHMIR PICKUP QUERY TEST")
    print("=" * 70)
    print(f"\nQuery: {query}\n")
    
    # Build graph
    graph = build_graph()
    
    # Create initial state
    initial_state = {
        "input": InputPayload(raw_text=query),
        "questions": Questions()
    }
    
    # Run workflow
    print("Running workflow...")
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
    print("\n" + "=" * 70)
    print("FINAL RESPONSE:")
    print("=" * 70)
    print(final_text)
    print("=" * 70)
    
    print(f"\nTrip Resolution:")
    print(f"  Trip ID: {trip_id}")
    print(f"  Confidence: {confidence}")
    print("=" * 70)
    
    return final_text

if __name__ == "__main__":
    conversation_messages = [
        "What kind of weather should I expect during the Andaman trip, and will there definitely be snowfall?",
        "What is the total cost of the trip?",
        "Can you tell me about the itinerary?",
    ]
    
    # Run conversation test
    test_conversation(conversation_messages)
    
    # Or run single query test
    # main()
