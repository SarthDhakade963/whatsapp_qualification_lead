import uuid


def generate_question_id() -> str:
    """Generate a unique question ID."""
    return f"q_{uuid.uuid4().hex[:8]}"


def generate_block_id() -> str:
    """Generate a unique block ID."""
    return f"block_{uuid.uuid4().hex[:8]}"

