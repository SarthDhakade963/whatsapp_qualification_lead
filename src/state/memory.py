# Conversation memory helpers
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from domain.trips.loader import get_all_trips


def _extract_topic_from_trip_id(trip_id: str) -> Optional[str]:
    """Extract topic name from trip_id (e.g., 'kashmir_zo_trip_TR-4Q7QMQQJ' -> 'kashmir')."""
    # Pattern: {topic}_zo_trip_TR-{code}
    if '_zo_trip_' in trip_id:
        topic = trip_id.split('_zo_trip_')[0]
        return topic
    return None


def trip_id_to_topic(trip_id: str) -> Optional[str]:
    """Map trip_id to topic name - auto-generated from all trips."""
    # Auto-generate mapping from trip_id pattern
    return _extract_topic_from_trip_id(trip_id)


def topic_to_trip_id(topic: str) -> Optional[str]:
    """Map topic name to trip_id - auto-generated from all trips."""
    # Auto-generate mapping from all discovered trips
    all_trips = get_all_trips()
    topic_lower = topic.lower()
    
    for trip_id in all_trips.keys():
        extracted_topic = _extract_topic_from_trip_id(trip_id)
        if extracted_topic and extracted_topic.lower() == topic_lower:
            return trip_id
    return None


class ConversationMemory:
    """Helper for managing conversation memory and authoritative state."""
    
    def __init__(self, store):
        self.store = store
    
    # ============================================================
    # MESSAGE HISTORY (for context/display)
    # ============================================================
    
    def get_history(self, session_id: str):
        """Get conversation history (messages) for a session."""
        return self.store.get(f"history:{session_id}") or []

    def add_message(self, session_id: str, message: dict):
        """Add a message to conversation history with timestamp."""
        history = self.get_history(session_id)
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat() + "Z"
        history.append(message)
        self.store.set(f"history:{session_id}", history)
    
    def get_recent_history(
        self, 
        session_id: str, 
        max_messages: int = 6,
        max_gap_hours: float = 36.0  # 1.5 days = 36 hours
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation history for graph context.
        Returns last 5-6 messages, but only if within time window (1.5-2 days).
        If there's a gap > max_gap_hours, starts fresh (returns empty).
        
        Args:
            session_id: Session identifier
            max_messages: Maximum number of messages to return (default: 6)
            max_gap_hours: Maximum time gap in hours before starting fresh (default: 36 = 1.5 days)
        
        Returns:
            List of recent messages with timestamps, filtered by time window
        """
        history = self.get_history(session_id)
        if not history:
            return []
        
        # Get most recent messages (last max_messages)
        recent_messages = history[-max_messages:] if len(history) > max_messages else history
        
        # Check time gaps - if gap > max_gap_hours, start fresh
        now = datetime.utcnow()
        filtered_messages = []
        
        for msg in reversed(recent_messages):  # Start from most recent
            msg_timestamp_str = msg.get("timestamp")
            if not msg_timestamp_str:
                # Old messages without timestamp - skip them (start fresh)
                break
            
            try:
                # Parse timestamp (handle both with and without timezone)
                msg_timestamp_str_clean = msg_timestamp_str.replace("Z", "")
                msg_timestamp = datetime.fromisoformat(msg_timestamp_str_clean)
                
                # Calculate time difference in hours
                time_diff = (now - msg_timestamp).total_seconds() / 3600
                
                # If gap is too large, stop here (start fresh from this point)
                if time_diff > max_gap_hours:
                    break
                
                filtered_messages.insert(0, msg)  # Insert at beginning to maintain order
                
            except (ValueError, AttributeError) as e:
                # Invalid timestamp format - skip this message
                continue
        
        return filtered_messages
    
    # ============================================================
    # AUTHORITATIVE CONVERSATION STATE
    # ============================================================
    
    def get_or_create_conversation_state(self, session_id: str) -> Dict[str, Any]:
        """
        Get or create authoritative conversation state.
        
        Returns conversation state matching this structure:
        {
            "conversation_id": "uuid",
            "version": 21,
            "intent_level": "browsing | evaluating | booking_ready",
            "risk_level": "low | medium | high",
            "momentum_state": "building | stalled | looping",
            "focus": {
                "primary_topic": "ladakh",
                "confidence": 0.64,
                "secondary": ["spiti"]
            },
            "anchor": {
                "topic": "spiti",
                "since_version": 14
            },
            "topic_decay": {
                "spiti": 0.35,
                "ladakh": 0.65
            },
            "handoff_status": "none | prepared | completed",
            "updated_at": "ISODate"
        }
        """
        state = self.store.get(f"state:{session_id}")
        if not state:
            # Create new conversation state
            state = self._default_state()
            state["conversation_id"] = str(uuid.uuid4())
            state["version"] = 0
            self.store.set(f"state:{session_id}", state)
        return state
    
    def update_conversation_state(
        self,
        session_id: str,
        trip_context: Optional[Dict[str, Any]] = None,
        interaction_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update authoritative conversation state with new decisions.
        Increments version and updates relevant fields.
        
        Args:
            session_id: Session identifier
            trip_context: {"trip_id": "...", "confidence": "LOW|MEDIUM|HIGH"}
            interaction_state: {"decision_stage": "...", "escalation_flag": bool}
        
        Returns:
            Updated conversation state
        """
        state = self.get_or_create_conversation_state(session_id)
        
        # Increment version
        state["version"] = state.get("version", 0) + 1
        
        # Update trip context and topic tracking
        if trip_context and trip_context.get("trip_id") and trip_context.get("trip_id") != "Not resolved":
            trip_id = trip_context["trip_id"]
            confidence_str = trip_context.get("confidence", "LOW")
            topic = trip_id_to_topic(trip_id)
            
            if topic:
                # Map confidence string to float (0.0 to 1.0)
                confidence_map = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 0.9}
                confidence = confidence_map.get(confidence_str, 0.5)
                
                # Get current focus
                current_focus = state.get("focus", {})
                current_primary = current_focus.get("primary_topic")
                
                # Update focus
                if current_primary != topic:
                    # Topic changed - update primary and secondary
                    secondary = current_focus.get("secondary", [])
                    
                    # Move old primary to secondary if it exists
                    if current_primary and current_primary not in secondary:
                        secondary = [current_primary] + secondary
                        # Keep max 2 secondary topics
                        secondary = secondary[:2]
                    
                    state["focus"] = {
                        "primary_topic": topic,
                        "confidence": confidence,
                        "secondary": secondary
                    }
                    
                    # Update anchor if this topic persists
                    anchor = state.get("anchor", {})
                    if anchor.get("topic") != topic:
                        state["anchor"] = {
                            "topic": topic,
                            "since_version": state["version"]
                        }
                else:
                    # Same topic - update confidence (take max)
                    state["focus"]["confidence"] = max(
                        state["focus"].get("confidence", 0),
                        confidence
                    )
                
                # Update topic decay
                # Decay all existing topics
                topic_decay = state.get("topic_decay", {})
                decay_factor = 0.9  # Decay by 10%
                for t in list(topic_decay.keys()):
                    topic_decay[t] = topic_decay[t] * decay_factor
                    # Remove topics with very low decay
                    if topic_decay[t] < 0.05:
                        del topic_decay[t]
                
                # Boost current topic
                current_decay = topic_decay.get(topic, 0)
                topic_decay[topic] = current_decay + (1 - current_decay) * 0.3
                
                # Normalize if sum gets too high
                total = sum(topic_decay.values())
                if total > 1.5:
                    topic_decay = {k: v / total * 1.2 for k, v in topic_decay.items()}
                
                state["topic_decay"] = topic_decay
        
        # Update interaction state and derived fields
        if interaction_state:
            decision_stage = interaction_state.get("decision_stage", "EVALUATING")
            escalation_flag = interaction_state.get("escalation_flag", False)
            
            # Update intent_level based on decision_stage
            if decision_stage == "ANSWERED":
                # If we're answering questions, user is evaluating
                state["intent_level"] = "evaluating"
            elif decision_stage == "ESCALATED":
                # Escalation might indicate browsing or need for help
                state["intent_level"] = "browsing"
            else:
                state["intent_level"] = "browsing"
            
            # Update risk_level based on escalation and confidence
            if escalation_flag:
                state["risk_level"] = "high"
            else:
                confidence = state.get("focus", {}).get("confidence", 0)
                if confidence < 0.4:
                    state["risk_level"] = "medium"
                else:
                    state["risk_level"] = "low"
            
            # Update handoff_status
            if escalation_flag:
                if state.get("handoff_status") == "none":
                    state["handoff_status"] = "prepared"
        
        # Update momentum_state
        version = state["version"]
        anchor = state.get("anchor", {})
        anchor_version = anchor.get("since_version", 0)
        anchor_topic = anchor.get("topic")
        primary_topic = state.get("focus", {}).get("primary_topic")
        
        if version < 3:
            state["momentum_state"] = "building"
        elif version - anchor_version > 5 and anchor_topic == primary_topic:
            # Same topic for many versions - might be stalled
            state["momentum_state"] = "stalled"
        elif len(state.get("topic_decay", {})) > 3:
            # Too many topics - might be looping
            state["momentum_state"] = "looping"
        else:
            state["momentum_state"] = "building"
        
        # Update timestamp
        state["updated_at"] = datetime.utcnow().isoformat() + "Z"
        
        # Save updated state
        self.store.set(f"state:{session_id}", state)
        
        return state
    
    def _default_state(self) -> Dict[str, Any]:
        """Get default conversation state structure."""
        return {
            "conversation_id": None,
            "version": 0,
            "intent_level": "browsing",
            "risk_level": "low",
            "momentum_state": "building",
            "focus": {
                "primary_topic": None,
                "confidence": 0.0,
                "secondary": []
            },
            "anchor": {
                "topic": None,
                "since_version": 0
            },
            "topic_decay": {},
            "handoff_status": "none",
            "updated_at": None
        }
