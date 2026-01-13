# Seat availability behavior
from typing import Optional, Dict, Any
from datetime import datetime
import re
from llm.client import LLMClient


def extract_date_from_text(text: str) -> Optional[str]:
    """
    Extract date from text in various formats.
    Returns date in YYYY-MM-DD format if found, None otherwise.
    """
    if not text:
        return None
    
    # Common date patterns
    # Pattern 1: "24th January 2026" or "24 January 2026"
    pattern1 = r'(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})'
    match = re.search(pattern1, text.lower())
    if match:
        day = match.group(1)
        month_name = match.group(2)
        year = match.group(3)
        
        month_map = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        month = month_map.get(month_name)
        if month:
            day_padded = day.zfill(2)
            return f"{year}-{month}-{day_padded}"
    
    # Pattern 2: "2026-01-24" or "2026/01/24"
    pattern2 = r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})'
    match = re.search(pattern2, text)
    if match:
        year = match.group(1)
        month = match.group(2).zfill(2)
        day = match.group(3).zfill(2)
        return f"{year}-{month}-{day}"
    
    return None


def format_date_for_display(date_str: str) -> str:
    """
    Format date string (YYYY-MM-DD) for display.
    Returns formatted string like "24th January 2026"
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        day = date_obj.day
        month_name = date_obj.strftime("%B")
        year = date_obj.year
        
        # Add ordinal suffix
        if 10 <= day % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        
        return f"{day}{suffix} {month_name} {year}"
    except (ValueError, TypeError):
        return date_str


def find_next_available_date(batches: list, after_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Find the next available batch with seats after a given date.
    If after_date is None, returns the first available batch.
    """
    if not batches:
        return None
    
    # Sort batches by start_date
    sorted_batches = sorted(batches, key=lambda x: x.get("start_date", ""))
    
    for batch in sorted_batches:
        start_date = batch.get("start_date")
        if not start_date:
            continue
        
        # If after_date is specified, skip batches before that date
        if after_date and start_date <= after_date:
            continue
        
        seats_left = batch.get("seats_left", "0")
        try:
            seats_int = int(seats_left) if seats_left != "<dynamic>" else 0
            if seats_int > 0:
                return batch
        except (ValueError, TypeError):
            # If seats_left is not a number, assume available
            return batch
    
    return None


def check_seat_availability_behavior(trip_data: Dict[str, Any], question_text: str) -> Optional[str]:
    """
    Check seat availability for a trip based on the question.
    Hybrid approach: Fast pattern matching first, LLM only for ambiguous cases.
    Returns appropriate response message or None if not a seat availability question.
    """
    if not trip_data or not question_text:
        return None
    
    text_lower = question_text.lower()
    
    # STEP 1: Fast pattern matching for CLEAR cases (no LLM needed)
    # This handles ~90% of cases instantly without LLM calls
    
    # Clear date question patterns - return None immediately (not seat availability)
    date_patterns = [
        r'\b(available\s+)?dates?\b',
        r'\bwhat\s+dates?\b',
        r'\bwhen\s+(is|does|will)\s+(the\s+)?(trip|journey|tour)\b',
        r'\bwhen\s+does\s+it\s+(start|begin)\b',
        r'\bschedule\b',
        r'\btiming\b',
        r'\bdeparture\s+date\b',
        r'\bstart\s+date\b',
        r'\bwhich\s+dates?\b',
    ]
    if any(re.search(pattern, text_lower) for pattern in date_patterns):
        return None  # Clearly a date question, skip LLM and batch checking
    
    # Clear seat availability patterns - proceed directly to batch checking
    clear_seat_patterns = [
        r'\b(seats?|seat\s+availability)\s+(available|left|remaining)\b',
        r'\b(are|is)\s+there\s+seats?\b',
        r'\b(do|does)\s+(you|we)\s+have\s+seats?\b',
        r'\bcan\s+i\s+book\b',
        r'\bis\s+it\s+available\s+to\s+book\b',
        r'\bseats?\s+left\b',
        r'\bseats?\s+remaining\b',
        r'\bhow\s+many\s+seats?\s+(are\s+)?(left|available|remaining)\b',
    ]
    is_clear_seat_question = any(re.search(pattern, text_lower) for pattern in clear_seat_patterns)
    
    # STEP 2: For ambiguous cases, use LLM to disambiguate
    if not is_clear_seat_question:
        # Check if it might be ambiguous (contains relevant keywords but unclear context)
        ambiguous_keywords = ["available", "book", "booking"]
        has_ambiguous_keywords = any(keyword in text_lower for keyword in ambiguous_keywords)
        
        if has_ambiguous_keywords:
            # Use LLM to disambiguate (only for ambiguous cases)
            llm = LLMClient()
            intent = llm.detect_intent(question_text)
            if intent != "SEAT_AVAILABILITY":
                return None
        else:
            # No relevant keywords at all, definitely not seat availability
            return None
    
    # STEP 3: Proceed with batch checking (same logic for both clear and LLM-confirmed cases)
    batches = trip_data.get("batches", {})
    available_batches = batches.get("available_batches", [])
    
    if not available_batches:
        return None
    
    # Extract date from question if mentioned
    mentioned_date = extract_date_from_text(question_text)
    
    if mentioned_date:
        # Check availability for the mentioned date
        for batch in available_batches:
            if batch.get("start_date") == mentioned_date:
                seats_left = batch.get("seats_left", "0")
                
                # Check if seats are available
                try:
                    seats_int = int(seats_left) if seats_left != "<dynamic>" else 0
                    if seats_int > 0:
                        return "Seats are available, book fast!"
                    else:
                        # Find next available date
                        next_available = find_next_available_date(available_batches, mentioned_date)
                        if next_available:
                            next_date_formatted = format_date_for_display(next_available["start_date"])
                            mentioned_date_formatted = format_date_for_display(mentioned_date)
                            return f"Unfortunately, we do not have seats on this date ({mentioned_date_formatted}), but we do have seats on next available date ({next_date_formatted})."
                        else:
                            mentioned_date_formatted = format_date_for_display(mentioned_date)
                            return f"Unfortunately, we do not have seats on this date ({mentioned_date_formatted})."
                except (ValueError, TypeError):
                    # If seats_left is not a number, assume available
                    return "Limited seats available — book soon to secure your spot."
        
        # Date mentioned but not found in batches
        mentioned_date_formatted = format_date_for_display(mentioned_date)
        next_available = find_next_available_date(available_batches)
        if next_available:
            next_date_formatted = format_date_for_display(next_available["start_date"])
            return f"Unfortunately, we do not have seats on this date ({mentioned_date_formatted}), but we do have seats on next available date ({next_date_formatted})."
        else:
            return f"Unfortunately, we do not have seats on this date ({mentioned_date_formatted})."
    else:
        # No specific date mentioned, check if any seats are available
        for batch in available_batches:
            seats_left = batch.get("seats_left", "0")
            try:
                seats_int = int(seats_left) if seats_left != "<dynamic>" else 0
                if seats_int > 0:
                    return "Seats are available, book fast!"
            except (ValueError, TypeError):
                # If seats_left is not a number, assume available
                return "Seats are available, book fast!"
        
        # No seats available in any batch
        next_available = find_next_available_date(available_batches)
        if next_available:
            next_date_formatted = format_date_for_display(next_available["start_date"])
            return f"Unfortunately, we do not have seats available right now, but we do have seats on next available date ({next_date_formatted})."
        else:
            return "Unfortunately, we do not have seats available right now."
    
    return None
