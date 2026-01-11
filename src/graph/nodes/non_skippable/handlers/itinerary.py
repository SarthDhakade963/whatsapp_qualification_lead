from typing import TypedDict, Dict, Any
from graph.state import HandlerOutput
from domain.trips.loader import get_trip_data
from llm.client import LLMClient
from utils.behaviors import check_empathetic_response, check_seat_availability


def itinerary_handler(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle itinerary questions using LLM-based fact extraction.
    Returns facts only.
    Updates answerable_processing.handler_outputs directly (for conditional/sequential execution)
    """
    answerable_processing = state.get("answerable_processing")
    if not answerable_processing or not answerable_processing.get("answer_plan"):
        return {}
    
    # Find itinerary blocks
    answer_plan = answerable_processing.get("answer_plan", {})
    itinerary_blocks = [
        block for block in answer_plan.get("answer_blocks", [])
        if block.get("handler") == "itinerary_handler"
    ]
    
    if not itinerary_blocks:
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
    
    # Process each itinerary block
    new_handler_outputs = []
    structured_questions = answerable_processing.get("structured_questions", [])
    
    for block in itinerary_blocks:
        # Get questions for this block
        question_ids = block.get("question_ids", [])
        questions = [
            q for q in structured_questions
            if isinstance(q, dict) and q.get("id") in question_ids
        ]
        
        # Extract facts using LLM
        facts = []
        for q in questions:
            question_text = q.get("text", "") if isinstance(q, dict) else getattr(q, "text", "")
            if question_text:
                # Check for seat availability first
                seat_availability_response = check_seat_availability(trip_data, question_text)
                if seat_availability_response:
                    facts.append(seat_availability_response)
                # Check for empathetic responses
                elif check_empathetic_response(question_text):
                    empathetic_response = check_empathetic_response(question_text)
                    facts.append(empathetic_response)
                else:
                    # Detect general "tell me about" questions and make them more specific
                    question_lower = question_text.lower()
                    is_general_question = any(phrase in question_lower for phrase in [
                        "tell me about", "what is", "describe", "tell about", "about the"
                    ])
                    
                    # For general questions, modify to extract summary information
                    if is_general_question and trip_data:
                        modified_question = "What is the description, duration, destination, itinerary highlights, and key features of this trip?"
                        extracted_facts = llm.extract_facts(modified_question, trip_data)
                    else:
                        extracted_facts = llm.extract_facts(question_text, trip_data)
                    
                    facts.extend(extracted_facts)
        
        # If no facts found, provide fallback message
        if not facts:
            if trip_data:
                facts = ["Itinerary details are available upon booking confirmation."]
            else:
                facts = ["I don't have information about a trip to that destination. Could you clarify which trip you're asking about (e.g., Kashmir, Andaman)?"]
        
        # Create handler output (as dict)
        output = {
            "block_id": block.get("block_id"),
            "facts": facts,
            "requires_confirmation": False
        }
        
        new_handler_outputs.append(output)
    
    # Update answerable_processing with new outputs (append to existing)
    answerable_processing = answerable_processing.copy()
    existing_outputs = answerable_processing.get("handler_outputs", [])
    if not existing_outputs:
        existing_outputs = []
    existing_outputs.extend(new_handler_outputs)
    answerable_processing["handler_outputs"] = existing_outputs
    
    return {"answerable_processing": answerable_processing}
