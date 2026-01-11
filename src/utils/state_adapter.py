"""Adapter functions to handle state as both dict and Pydantic models."""

from typing import Any, Dict, Optional


def get_state_value(state: Any, key: str, default: Any = None) -> Any:
    """Get value from state whether it's a dict or Pydantic model."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def set_state_value(state: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    """Set value in state dict."""
    state = state.copy() if isinstance(state, dict) else {}
    state[key] = value
    return state


def get_nested_attr(obj: Any, attr_path: str, default: Any = None) -> Any:
    """Get nested attribute whether it's a dict or Pydantic model."""
    attrs = attr_path.split(".")
    current = obj
    
    for attr in attrs:
        if isinstance(current, dict):
            current = current.get(attr)
        else:
            current = getattr(current, attr, None)
        
        if current is None:
            return default
    
    return current


def update_nested_dict(obj: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    """Update nested dict path."""
    obj = obj.copy()
    parts = path.split(".")
    current = obj
    
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    
    current[parts[-1]] = value
    return obj


def to_dict(obj: Any) -> Dict[str, Any]:
    """Convert object to dict (Pydantic model or already dict)."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)
