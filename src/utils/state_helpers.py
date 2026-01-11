"""Helper functions for working with state (dict or Pydantic models)."""

from typing import Any, Dict


def get_state_value(state: Any, key: str, default: Any = None) -> Any:
    """Get value from state whether it's a dict or Pydantic model."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def set_state_value(state: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    """Set value in state dict and return updated dict."""
    state[key] = value
    return state


def get_nested_value(obj: Any, attr: str, default: Any = None) -> Any:
    """Get nested attribute whether it's a dict or Pydantic model."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)
