from typing import TypedDict, Dict, Any
import re
from graph.state import MergedOutput
from utils.state_adapter import get_state_value, to_dict
from utils.behaviors import check_decision_confirmation, check_call_request


def merge_outputs(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge outputs from answerable and skippable branches.
    Only modifies: merged_output
    """
    # Check for booking confirmation first
    input_obj = get_state_value(state, "input", {})
    raw_text = input_obj.get("raw_text") if isinstance(input_obj, dict) else getattr(input_obj, "raw_text", "")
    
    if raw_text:
        text_lower = raw_text.lower()
        
        # Check if it's a QUESTION (should NOT trigger booking confirmation)
        question_patterns = [
            r'\b(what|how|when|where|why|which|who)\s+',
            r'\b(can\s+you|could\s+you|will\s+you|would\s+you|should\s+i|can\s+i|will\s+i|would\s+i)\s+',
            r'\b(tell\s+me|explain|describe|share|show)\s+',
            r'\bis\s+there|are\s+there|do\s+you|does\s+it',
            r'\?\s*$',  # Ends with question mark
        ]
        is_question = any(re.search(pattern, text_lower) for pattern in question_patterns)
        
        # Check if it's a HYPOTHETICAL question (should NOT trigger booking confirmation)
        hypothetical_patterns = [
            r'\bwhat\s+if\b',
            r'\bif\s+i\b',
            r'\bwill\s+i\s+get\b',
            r'\bwould\s+i\s+get\b',
            r'\bcan\s+i\s+get\b',
            r'\bshould\s+i\s+get\b',
            r'\bif\s+i\s+book\b',
            r'\bwhen\s+i\s+book\b',
            r'\bafter\s+i\s+book\b',
        ]
        is_hypothetical = any(re.search(pattern, text_lower) for pattern in hypothetical_patterns)
        
        # Only check for actual booking confirmation if it's NOT a question and NOT hypothetical
        has_booking_confirmation = False
        if not is_question and not is_hypothetical:
            # Patterns that indicate ACTUAL booking (past tense, completed actions, statements)
            actual_booking_patterns = [
                r'\b(i|i\'ve|i have)\s+(just\s+)?booked\b',
                r'\bjust\s+booked\b',
                r'\bbooked\s+(it|the\s+trip|the\s+package)\b',
                r'\bdone\s+booking\b',
                r'\bcompleted\s+booking\b',
                r'\bbooking\s+is\s+confirmed\b',
                r'\bpayment\s+is\s+done\b',
                r'\bpayment\s+completed\b',
                r'\bi\s+paid\b',
                r'\bpayment\s+made\b',
            ]
            has_booking_confirmation = any(re.search(pattern, text_lower) for pattern in actual_booking_patterns)
        
        # Detect concern/complaint keywords
        concern_keywords = [
            "no update", "no response", "no reply", "no information", 
            "issue", "problem", "concern", "complaint", "not received",
            "haven't received", "didn't get", "missing", "wrong", "error"
        ]
        has_concern = any(keyword in text_lower for keyword in concern_keywords)
        
        # If booking confirmation but no concerns, respond with celebration
        if has_booking_confirmation and not has_concern:
            return {"merged_output": {"final_text": "Zo Zo 😍"}}
        # If both booking confirmation and concerns, let normal flow handle it
        # (don't return early, continue to process the concern)
        
        # Check for call requests (priority - before processing answerable/skippable)
        conversation_history = state.get("conversation_history", [])
        call_response = check_call_request(raw_text, conversation_history)
        if call_response:
            # If it's a follow-up, also set escalation flag
            if call_response.get("is_followup"):
                # Update interaction state to escalate
                # The summary will be in call_response["summary"]
                return {
                    "merged_output": {"final_text": call_response["response"]},
                    "interaction_state": {
                        "decision_stage": "ESCALATED",
                        "escalation_flag": True,
                        "call_summary": call_response.get("summary", "")
                    }
                }
            else:
                return {"merged_output": {"final_text": call_response["response"]}}
    
    parts = []
    
    # Add answerable answer if present
    answerable_processing = get_state_value(state, "answerable_processing")
    if answerable_processing:
        answerable_dict = to_dict(answerable_processing)
        answer_text = answerable_dict.get("answer_text")
        if answer_text:
            parts.append(answer_text)
    
    # Add skippable boundaries ONLY if there are forbidden questions in CURRENT batch
    # Check if there are forbidden questions in the current batch
    questions = get_state_value(state, "questions", {})
    questions_dict = to_dict(questions)
    partitioned = questions_dict.get("partitioned", {})
    forbidden_in_current_batch = partitioned.get("skippable", {}).get("forbidden", [])
    
    # Only add boundaries if there are forbidden questions in current batch
    if forbidden_in_current_batch:
        skippable_actions = get_state_value(state, "skippable_actions")
        if skippable_actions:
            skippable_dict = to_dict(skippable_actions)
            boundaries = skippable_dict.get("boundaries", [])
            if boundaries:
                parts.extend(boundaries)
    
    # Combine parts
    final_text = " ".join(parts).strip()
    
    # Check for decision/confirmation statements before generic fallback
    if not final_text and raw_text:
        decision_response = check_decision_confirmation(raw_text)
        if decision_response:
            return {"merged_output": {"final_text": decision_response}}
    
    if not final_text:
        final_text = "I'm here to help. Could you rephrase your question?"
    
    merged_output = {
        "final_text": final_text
    }
    
    return {"merged_output": merged_output}
