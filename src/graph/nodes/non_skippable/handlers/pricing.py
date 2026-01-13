from typing import TypedDict, Dict, Any
from graph.state import HandlerOutput
from domain.trips.loader import get_trip_data
from domain.policies import REFUND_POLICY, DISCOUNT_POLICY
from llm.client import LLMClient
from utils.behaviors import check_empathetic_response, check_seat_availability


def pricing_handler(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle pricing and policy questions using LLM-based fact extraction.
    Returns facts only.
    Updates answerable_processing.handler_outputs directly (for conditional/sequential execution)
    """
    answerable_processing = state.get("answerable_processing")
    if not answerable_processing or not answerable_processing.get("answer_plan"):
        return {}
    
    # Find pricing blocks
    answer_plan = answerable_processing.get("answer_plan", {})
    pricing_blocks = [
        block for block in answer_plan.get("answer_blocks", [])
        if block.get("handler") == "pricing_handler"
    ]
    
    if not pricing_blocks:
        return {}
    
    # Get trip data based on trip_context
    trip_context = answerable_processing.get("trip_context", {})
    trip_id = trip_context.get("trip_id", "") if isinstance(trip_context, dict) else getattr(trip_context, "trip_id", "")
    
    if not trip_id:
        trip_id = ""
    
    trip_data = get_trip_data(trip_id)
    if not trip_data or not isinstance(trip_data, dict):
        trip_data = {}
    
    # Initialize LLM client for fact extraction
    llm = LLMClient()
    
    # Process each pricing block
    new_handler_outputs = []
    structured_questions = answerable_processing.get("structured_questions", [])
    
    for block in pricing_blocks:
        question_ids = block.get("question_ids", [])
        questions = [
            q for q in structured_questions
            if isinstance(q, dict) and q.get("id") in question_ids
        ]
        
        facts = []
        for q in questions:
            question_text = q.get("text", "") if isinstance(q, dict) else getattr(q, "text", "")
            text_lower = question_text.lower() if question_text else ""
            
            # Special handling for refund/cancellation policy questions
            # Distinguish between informational questions and guarantee requests
            if "refund" in text_lower or "cancellation" in text_lower:
                # If asking about policy details (informational), provide policy
                if any(phrase in text_lower for phrase in ["what is", "tell me", "explain", "policy", "cancellation policy"]):
                    facts.append(REFUND_POLICY["full_policy_text"])
                else:
                    # If asking for refund/guarantee (decision), use boundary message
                    facts.append(REFUND_POLICY["boundary_message"])
            # Special handling for discount/offer questions - use policy boundary message
            elif any(keyword in text_lower for keyword in ["discount", "offer", "deal", "promo", "coupon", "cheaper", "lower price", "best price", "first time"]):
                facts.append(DISCOUNT_POLICY["boundary_message"])
            elif question_text:
                # Check for seat availability first
                seat_availability_response = check_seat_availability(trip_data, question_text)
                if seat_availability_response:
                    facts.append(seat_availability_response)
                # Check for empathetic responses
                elif check_empathetic_response(question_text):
                    empathetic_response = check_empathetic_response(question_text)
                    facts.append(empathetic_response)
                else:
                    # Use LLM to extract facts for other pricing questions
                    extracted_facts = llm.extract_facts(question_text, trip_data)
                    facts.extend(extracted_facts)
        
        # If no facts found, provide fallback message
        if not facts:
            if trip_data:
                facts = ["I'd be happy to share pricing details. Would you like to know about the trip cost, payment options, or booking information?"]
            else:
                facts = ["I'd be happy to share that information. Could you clarify which trip you're asking about (e.g., Kashmir, Andaman)?"]
        
        # Create handler output (as dict)
        output = {
            "block_id": block.get("block_id"),
            "facts": facts,
            "requires_confirmation": False
        }
        
        new_handler_outputs.append(output)
    
    answerable_processing = answerable_processing.copy()
    existing_outputs = answerable_processing.get("handler_outputs", [])
    if not existing_outputs:
        existing_outputs = []
    existing_outputs.extend(new_handler_outputs)
    answerable_processing["handler_outputs"] = existing_outputs
    
    return {"answerable_processing": answerable_processing}
