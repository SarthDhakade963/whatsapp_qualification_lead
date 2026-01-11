from typing import TypedDict, Dict, Any
from graph.state import TripContext
from utils.state_adapter import get_state_value, to_dict
from domain.trips.loader import get_all_trips, get_trip_data
from state.memory import topic_to_trip_id


def _generate_trip_keywords(trip_data: Dict) -> list:
    """Auto-generate keywords from trip data."""
    keywords = []
    
    # Extract from destination
    destination = trip_data.get("destination", "").lower()
    if destination:
        # Split by comma and take first part (main location)
        main_location = destination.split(",")[0].strip()
        keywords.append(main_location)
        # Also add individual words from main_location
        if " " in main_location:
            main_location_words = main_location.split()
            for word in main_location_words:
                if len(word) > 3 and word not in keywords:
                    keywords.append(word)
        # Add full destination if multi-word
        if " " in destination:
            keywords.append(destination)
    
    # Extract from name
    name = trip_data.get("name", "").lower()
    if name:
        # Remove common prefixes like "Experience"
        name_clean = name.replace("experience", "").replace("winter edition", "").replace("(", "").replace(")", "").strip()
        if name_clean:
            keywords.append(name_clean)
            # Also add individual words from name
            name_words = name_clean.split()
            for word in name_words:
                if len(word) > 3 and word not in keywords:
                    keywords.append(word)
    
    # Extract from itinerary highlights
    highlights = trip_data.get("itinerary_highlights", [])
    for highlight in highlights:
        if isinstance(highlight, str):
            # Extract location names (before colon)
            if ":" in highlight:
                location = highlight.split(":")[0].strip().lower()
                if location and location not in keywords:
                    keywords.append(location)
            # Also extract key words from highlight
            words = highlight.lower().split()
            for word in words:
                # Remove punctuation
                word_clean = word.strip(".,;:()[]{}")
                if len(word_clean) > 3 and word_clean not in keywords and word_clean not in ["and", "the", "for", "with", "from", "through"]:
                    if any(char.isalpha() for char in word_clean):
                        keywords.append(word_clean)
    
    # Extract from logistics meeting point
    logistics = trip_data.get("logistics", {})
    meeting_point = logistics.get("meeting_point", "").lower()
    if meeting_point:
        # Extract city name (before parenthesis)
        city = meeting_point.split("(")[0].strip()
        if city and city not in keywords:
            keywords.append(city)
    
    return list(set(keywords))  # Remove duplicates


def resolve_trip_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve trip context from conversation using keyword matching.
    Only modifies: answerable_processing.trip_context
    """
    answerable_processing = get_state_value(state, "answerable_processing")
    if not answerable_processing:
        return {}
    
    answerable_dict = to_dict(answerable_processing)
    normalized_text = answerable_dict.get("normalized_text", "")
    structured_questions = answerable_dict.get("structured_questions", [])
    
    # Combine text for analysis
    question_texts = [
        q.get("text", "") if isinstance(q, dict) else getattr(q, "text", "")
        for q in structured_questions
    ]
    combined_text = (normalized_text + " " + " ".join(question_texts)).lower()
    
    # Get conversation history if available and add to combined text for context
    conversation_history = state.get("conversation_history", [])
    if conversation_history:
        # Extract previous user messages from history
        previous_messages = [
            msg.get("content", "").lower() 
            for msg in conversation_history 
            if isinstance(msg, dict) and msg.get("role") == "user"
        ]
        # Add previous messages to combined text for context (most recent first)
        if previous_messages:
            combined_text = " ".join(previous_messages) + " " + combined_text
    
    # Keyword-based matching
    trip_id = None
    confidence = "LOW"
    
    # Auto-generate trip keywords from trip data
    all_trips = get_all_trips()
    trip_keywords = {}
    for tid, trip_data in all_trips.items():
        trip_keywords[tid] = _generate_trip_keywords(trip_data)
    
    # Score each trip
    trip_scores = {}
    
    for tid, keywords in trip_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in combined_text:
                score += 2 if len(keyword.split()) > 1 else 1  # Multi-word keywords get higher score
        
        # Check trip name
        trip_data = all_trips.get(tid)
        if trip_data:
            trip_name = trip_data.get("name", "").lower()
            if trip_name in combined_text:
                score += 5
        
        if score > 0:
            trip_scores[tid] = score
    
    # Get best matching trip
    if trip_scores:
        best_trip_id = max(trip_scores.items(), key=lambda x: x[1])[0]
        best_score = trip_scores[best_trip_id]
        trip_id = best_trip_id
        
        # Set confidence based on score
        if best_score >= 5:
            confidence = "HIGH"
        elif best_score >= 3:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
    
    # Fallback to authoritative conversation state if no trip found in current message
    if not trip_id:
        conversation_state = state.get("conversation_state", {})
        if conversation_state:
            focus = conversation_state.get("focus", {})
            primary_topic = focus.get("primary_topic")
            topic_confidence = focus.get("confidence", 0.0)
            
            if primary_topic:
                # Use authoritative state's primary topic
                fallback_trip_id = topic_to_trip_id(primary_topic)
                if fallback_trip_id:
                    trip_id = fallback_trip_id
                    # Map confidence float (0.0-1.0) to string
                    if topic_confidence >= 0.7:
                        confidence = "HIGH"
                    elif topic_confidence >= 0.4:
                        confidence = "MEDIUM"
                    else:
                        confidence = "LOW"
                    # Lower confidence since we're using fallback
                    if confidence == "HIGH":
                        confidence = "MEDIUM"
                    elif confidence == "MEDIUM":
                        confidence = "LOW"
    
    # Final default fallback
    if not trip_id:
        trip_id = "kashmir_zo_trip_TR-4Q7QMQQJ"
        confidence = "LOW"
    
    trip_context = {
        "trip_id": trip_id,
        "confidence": confidence
    }
    
    # Update answerable_processing
    answerable_processing = answerable_dict.copy()
    answerable_processing["trip_context"] = trip_context
    
    return {"answerable_processing": answerable_processing}
