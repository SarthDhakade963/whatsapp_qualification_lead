#!/usr/bin/env python3
"""Streamlit chat interface for testing WhatsApp conversation flow."""

import sys
import os
import streamlit as st

# Add src to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_current_dir, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Prevent torch loading issues
os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")

# Configure LangSmith BEFORE importing graph modules
from app.settings import Settings
settings = Settings()

if settings.langsmith_tracing:
    langsmith_key = settings.effective_langsmith_api_key()
    if langsmith_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_PROJECT"] = settings.effective_langsmith_project()
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
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

# NOW import graph modules
from graph.build_graph import build_graph
from graph.state import InputPayload, Questions
from state.store import StateStore
from state.memory import ConversationMemory

# Page config
st.set_page_config(
    page_title="WhatsApp Lead Qualification Chat",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = None
if "memory" not in st.session_state:
    st.session_state.memory = None
if "store" not in st.session_state:
    st.session_state.store = None
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_session_123"

# Initialize components
@st.cache_resource
def initialize_graph():
    """Initialize the graph once and cache it."""
    return build_graph()

@st.cache_resource
def initialize_memory():
    """Initialize memory store once and cache it."""
    store = StateStore()
    return ConversationMemory(store)

if st.session_state.graph is None:
    with st.spinner("Initializing conversation system..."):
        st.session_state.graph = initialize_graph()
        st.session_state.memory = initialize_memory()
        st.session_state.store = StateStore()

# Sidebar for controls and info
with st.sidebar:
    st.title("⚙️ Controls")
    
    # Session ID input
    session_id = st.text_input("Session ID", value=st.session_state.session_id)
    st.session_state.session_id = session_id
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", type="primary"):
        st.session_state.messages = []
        # Clear memory for this session
        if st.session_state.memory:
            # Reset conversation state
            st.session_state.memory.store.set(f"history:{session_id}", [])
            st.session_state.memory.store.set(f"conversation_state:{session_id}", None)
        st.rerun()
    
    st.divider()
    
    # Conversation state info
    st.subheader("📊 Conversation State")
    if st.session_state.memory:
        conversation_state = st.session_state.memory.get_or_create_conversation_state(session_id)
        history = st.session_state.memory.get_history(session_id)
        recent_history = st.session_state.memory.get_recent_history(session_id, max_messages=6, max_gap_hours=36.0)
        
        st.metric("Total Messages", len(history))
        st.metric("Recent Messages", len(recent_history))
        st.metric("State Version", conversation_state.get('version', 0))
        
        focus = conversation_state.get('focus', {})
        st.write(f"**Focus:** {focus.get('primary_topic', 'None')}")
        st.write(f"**Confidence:** {focus.get('confidence', 0):.2f}")
        st.write(f"**Intent:** {conversation_state.get('intent_level', 'browsing')}")
        st.write(f"**Risk:** {conversation_state.get('risk_level', 'low')}")

# Main chat interface
st.title("💬 WhatsApp Lead Qualification Chat")
st.caption("Test the conversation flow with various messages")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("📊 Response Metadata"):
                metadata = message["metadata"]
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Trip ID:** {metadata.get('trip_id', 'N/A')}")
                    st.write(f"**Confidence:** {metadata.get('confidence', 'N/A')}")
                with col2:
                    st.write(f"**Decision Stage:** {metadata.get('decision_stage', 'N/A')}")
                    st.write(f"**Escalation:** {metadata.get('escalation_flag', False)}")

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process message
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                # Load conversation state
                recent_history = st.session_state.memory.get_recent_history(
                    session_id, max_messages=6, max_gap_hours=36.0
                )
                conversation_state = st.session_state.memory.get_or_create_conversation_state(session_id)
                
                # Initialize state
                initial_state = {
                    "input": InputPayload(raw_text=prompt),
                    "questions": Questions(),
                    "conversation_history": recent_history if recent_history else None,
                    "conversation_state": conversation_state
                }
                
                # Run graph
                final_state = st.session_state.graph.invoke(initial_state)
                
                # Extract response
                merged_output = final_state.get("merged_output", {})
                response_text = merged_output.get("final_text", "No output generated")
                
                # Extract metadata
                answerable_processing = final_state.get("answerable_processing", {})
                trip_context = answerable_processing.get("trip_context", {}) if answerable_processing else {}
                interaction_state = final_state.get("interaction_state", {})
                
                metadata = {
                    "trip_id": trip_context.get("trip_id", "Not resolved") if trip_context else "Not resolved",
                    "confidence": trip_context.get("confidence", "N/A") if trip_context else "N/A",
                    "decision_stage": interaction_state.get("decision_stage", "N/A") if interaction_state else "N/A",
                    "escalation_flag": interaction_state.get("escalation_flag", False) if interaction_state else False
                }
                
                # Display response
                st.markdown(response_text)
                
                # Save to conversation
                st.session_state.memory.add_message(session_id, {
                    "role": "user",
                    "content": prompt
                })
                st.session_state.memory.add_message(session_id, {
                    "role": "assistant",
                    "content": response_text
                })
                
                # Update conversation state
                st.session_state.memory.update_conversation_state(
                    session_id,
                    trip_context=trip_context if metadata["trip_id"] != "Not resolved" else None,
                    interaction_state=interaction_state if interaction_state else None
                )
                
                # Add to messages with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "metadata": metadata
                })
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
    
    st.rerun()