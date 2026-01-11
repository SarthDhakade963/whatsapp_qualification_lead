from typing import Optional, Dict, Any, List
from domain.behaviors import EMPATHETIC_RESPONSES, check_seat_availability_behavior
import re


def check_empathetic_response(question_text: str) -> Optional[str]:
    """
    Check if a question matches any empathetic response pattern.
    Returns the response if matched, None otherwise.
    """
    if not question_text:
        return None
    
    text_lower = question_text.lower()
    
    for behavior_type, behavior_data in EMPATHETIC_RESPONSES.items():
        keywords = behavior_data.get("keywords", [])
        if any(keyword in text_lower for keyword in keywords):
            return behavior_data.get("response")
    
    return None


def check_seat_availability(trip_data: Dict[str, Any], question_text: str) -> Optional[str]:
    """
    Check seat availability behavior.
    Returns appropriate response message or None if not a seat availability question.
    """
    return check_seat_availability_behavior(trip_data, question_text)


def check_decision_confirmation(text: str) -> Optional[str]:
    """
    Check if text is a decision/confirmation statement.
    Returns appropriate response message or None if not a decision statement.
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Check for decision/confirmation keywords
    decision_keywords = [
        "will confirm", "confirm after", "let me think", "need time",
        "will decide", "decide later", "get back", "think about it",
        "consider", "will let you know", "confirm later", "after some time",
        "i'll confirm", "i will confirm", "confirm in", "after days"
    ]
    
    if any(keyword in text_lower for keyword in decision_keywords):
        return "No problem! Take your time. Feel free to reach out when you're ready to book or if you have any questions."
    
    return None


def check_call_request(text: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """
    Check if text is a call request.
    Returns dict with "response", "is_followup", and optionally "summary" keys, or None if not a call request.
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Check for call request keywords
    call_keywords = [
        "call", "quick call", "get on a call", "can we call", "schedule a call",
        "arrange a call", "phone call", "video call", "want to call"
    ]
    
    is_call_request = any(keyword in text_lower for keyword in call_keywords)
    
    if not is_call_request:
        return None
    
    # Check if this is a follow-up to the call request question
    is_followup = False
    if conversation_history:
        # Check if last assistant message was the call follow-up question
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "").lower()
                if "arrange a call" in content or "what you'd like to discuss on the call" in content or "preferred time" in content:
                    is_followup = True
                    break
            elif msg.get("role") == "user":
                # If we hit a user message before finding the follow-up question, it's not a follow-up
                break
    
    if is_followup:
        # Extract availability time and discussion points from the current user message
        availability_time = None
        discussion_points = text.strip()
        availability_keywords = [
            "available", "free", "time", "when", "call me", "reach me",
            "morning", "afternoon", "evening", "night", "am", "pm",
            "today", "tomorrow", "week", "weekend", "between", "after", "before", "now"
        ]
        
        # Check if user mentioned availability in current text
        if any(keyword in text_lower for keyword in availability_keywords):
            # Try to extract time-related phrases
            # Look for time patterns like "10 am", "2 pm", "evening", "morning", "call me now", etc.
            time_patterns = [
                r'call\s+me\s+now',
                r'\d{1,2}\s*(am|pm|AM|PM)',
                r'(morning|afternoon|evening|night)',
                r'(today|tomorrow)',
                r'between\s+\d{1,2}\s*(and|to|-)\s*\d{1,2}',
                r'after\s+\d{1,2}',
                r'before\s+\d{1,2}',
                r'\bnow\b'
            ]
            for pattern in time_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    availability_time = text[match.start():match.end()].strip()
                    # Remove availability time from discussion points
                    discussion_points = (text[:match.start()] + text[match.end():]).strip()
                    discussion_points = re.sub(r'\s+', ' ', discussion_points).strip()
                    # Clean up trailing punctuation
                    discussion_points = re.sub(r'[.!?]+$', '', discussion_points).strip()
                    break
            
            # If no pattern matched but availability keywords are present, try sentence-level extraction
            if not availability_time:
                sentences = re.split(r'[.!?]', text)
                availability_sentences = []
                discussion_sentences = []
                for sentence in sentences:
                    sentence_lower = sentence.lower().strip()
                    if any(keyword in sentence_lower for keyword in ["call me", "reach me", "available", "free", "time"]):
                        if not availability_time:
                            availability_time = sentence.strip()
                        availability_sentences.append(sentence.strip())
                    else:
                        if sentence.strip():
                            discussion_sentences.append(sentence.strip())
                
                # Reconstruct discussion points without availability sentences
                if discussion_sentences:
                    discussion_points = ". ".join(discussion_sentences).strip()
                elif not availability_time:
                    # If still no match, use the full text as availability info
                    availability_time = text.strip()
                    discussion_points = ""
        
        # Generate summary from conversation history
        summary_parts = []
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user" and content:
                    summary_parts.append(f"User: {content}")
        
        # Add current message's discussion points to summary (if not empty)
        if discussion_points:
            summary_parts.append(f"User (Current): {discussion_points}")
        elif not summary_parts:
            # If no discussion points extracted and no history, use full message
            summary_parts.append(f"User (Current): {text}")
        
        conversation_summary = "\n".join(summary_parts) if summary_parts else "No conversation history available."
        
        # Build enhanced summary with availability time and discussion points
        if availability_time:
            summary = f"📞 CALL REQUEST SUMMARY\n\n💬 Discussion Points:\n{conversation_summary}\n\n⏰ Preferred Call Time: {availability_time}\n\nOur team will reach out to you at your preferred time."
        else:
            summary = f"📞 CALL REQUEST SUMMARY\n\n💬 Discussion Points:\n{conversation_summary}\n\n⏰ Preferred Call Time: Not specified"
        
        return {
            "response": "Perfect! I've noted your request.\n\nOur team will reach out to you at your preferred time.",
            "is_followup": True,
            "summary": summary
        }
    else:
        # First call request - ask follow-up question with availability time
        return {
            "response": "I can help with that 🙂\nBefore I arrange a call, could you briefly share:\n1. What you'd like to discuss on the call?\n2. Your preferred time/availability for the call?",
            "is_followup": False
        }
