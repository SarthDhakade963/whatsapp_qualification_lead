# Load prompt templates from files
import os

_PROMPT_DIR = os.path.dirname(__file__)


def _load_prompt(filename: str) -> str:
    """Load prompt template from file."""
    filepath = os.path.join(_PROMPT_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback to simple prompts if files don't exist
        return ""


CLASSIFIER_PROMPT = _load_prompt("classifier.txt") or """Classify the following question into EXACTLY ONE of these categories: ANSWERABLE, FORBIDDEN, MALFORMED, HOSTILE.

Rules:
- ANSWERABLE: Questions we can answer with our trip information
- FORBIDDEN: Questions about refunds, guarantees, or policies we cannot discuss directly
- MALFORMED: Questions that are too short, unclear, or nonsensical
- HOSTILE: Questions with hostile, offensive, or inappropriate language

Question: {question_text}

Return ONLY the classification word (one of: ANSWERABLE, FORBIDDEN, MALFORMED, HOSTILE). No explanation, just the word."""

PLANNER_PROMPT = _load_prompt("planner.txt") or """Create an answer plan that groups questions by category and assigns appropriate handlers.

Structured Questions:
{structured_questions}

Trip Context:
{trip_context}

Return a JSON object with answer_blocks."""

COMPOSER_PROMPT = _load_prompt("composer.txt") or """Compose a natural language answer from the provided facts.

Handler Outputs: {handler_outputs}
Original Question: {normalized_text}

Return a coherent, natural answer."""

CATEGORIZER_PROMPT = _load_prompt("categorizer.txt") or """Categorize the following question into EXACTLY ONE of these categories: LOGISTICS, COST, ITINERARY, POLICY.

Rules:
- LOGISTICS: Questions about pickup points, transportation, accommodation, hotels, travel arrangements
- COST: Questions about pricing, costs, fees, payment, budget, expenses
- ITINERARY: Questions about schedule, daily activities, places to visit, day-by-day plan
- POLICY: Questions about refund policies, cancellation policies, terms and conditions

Question: {question_text}

Return ONLY the category word (one of: LOGISTICS, COST, ITINERARY, POLICY). No explanation, just the word."""

EXTRACTOR_PROMPT = _load_prompt("extractor.txt") or """You are a fact extraction system for a travel booking assistant.

Question: {question_text}

Trip Data (JSON):
{trip_data}

Instructions:
- Extract ONLY relevant facts from the trip data that answer the question
- Return facts as a JSON array of strings
- Each fact should be a complete, standalone statement
- If the question asks about something not in trip data, return an empty array
- Do not make up information not present in trip data
- Be specific and accurate
- Use natural language for facts (not just raw data values)

Return ONLY a JSON array of fact strings, no explanations. Example: ["Fact 1", "Fact 2"]"""

INTENT_DETECTOR_PROMPT = _load_prompt("intent_detector.txt") or """Determine the intent of the following question about a travel trip.

Question: {question_text}

Intent Categories:
- SEAT_AVAILABILITY: Questions asking about seat availability, whether seats are available, booking availability, or if they can book now
- DATES: Questions asking about trip dates, available dates, schedule, when the trip happens, departure dates
- OTHER: Any other question that doesn't fit the above categories

Return ONLY the intent category (one of: SEAT_AVAILABILITY, DATES, OTHER). No explanation, just the word."""

__all__ = ["CLASSIFIER_PROMPT", "PLANNER_PROMPT", "COMPOSER_PROMPT", "CATEGORIZER_PROMPT", "EXTRACTOR_PROMPT", "INTENT_DETECTOR_PROMPT"]
