import re


def normalize_text(text: str) -> str:
    """Normalize input text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def split_into_questions(text: str) -> list[str]:
    """Split text into atomic questions based on conjunctions and question marks."""
    # Split on common conjunctions and question marks
    questions = re.split(r'[?]| and | what about | how about ', text, flags=re.IGNORECASE)
    questions = [q.strip() for q in questions if q.strip()]
    
    # If no clear split, treat as single question
    if not questions:
        questions = [text]
    
    return questions

