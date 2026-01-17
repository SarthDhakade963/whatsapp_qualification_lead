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
    Updates answerable_processing.handler_outputs directly.
    
    Note: Executes in parallel with other handlers. If no pricing blocks exist,
    returns {} immediately (early exit) for optimal performance.
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
        
        # Separate questions by processing type
        facts_map = {}  # Map question_text to facts list
        llm_questions = []  # Questions that need LLM extraction
        
        for q in questions:
            question_text = q.get("text", "") if isinstance(q, dict) else getattr(q, "text", "")
            if not question_text:
                continue
            
            text_lower = question_text.lower()
            
            # Special handling for refund/cancellation policy questions
            # Distinguish between informational questions and guarantee requests
            if "refund" in text_lower or "cancellation" in text_lower:
                # If asking about policy details (informational), provide policy
                if any(phrase in text_lower for phrase in ["what is", "tell me", "explain", "policy", "cancellation policy"]):
                    facts_map[question_text] = [REFUND_POLICY["full_policy_text"]]
                else:
                    # If asking for refund/guarantee (decision), use boundary message
                    facts_map[question_text] = [REFUND_POLICY["boundary_message"]]
                continue
            
            # Special handling for discount/offer questions - use policy boundary message
            if any(keyword in text_lower for keyword in ["discount", "offer", "deal", "promo", "coupon", "cheaper", "lower price", "best price", "first time"]):
                facts_map[question_text] = [DISCOUNT_POLICY["boundary_message"]]
                continue
            
            # Check for seat availability first
            seat_availability_response = check_seat_availability(trip_data, question_text)
            if seat_availability_response:
                facts_map[question_text] = [seat_availability_response]
                continue
            
            # Check for empathetic responses
            empathetic_response = check_empathetic_response(question_text)
            if empathetic_response:
                facts_map[question_text] = [empathetic_response]
                continue
            
            # Add to LLM batch
            llm_questions.append(question_text)
        
        # BATCH: Extract facts for all LLM questions in a single call
        if llm_questions:
            batch_results = llm.extract_facts_batch(llm_questions, trip_data)
            
            # Map batch results back to questions
            for q_text, q_facts in batch_results.items():
                if q_text not in facts_map:
                    facts_map[q_text] = []
                facts_map[q_text].extend(q_facts)
        
        # Combine all facts from all questions
        facts = []
        for q in questions:
            question_text = q.get("text", "") if isinstance(q, dict) else getattr(q, "text", "")
            if question_text and question_text in facts_map:
                facts.extend(facts_map[question_text])
        
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
