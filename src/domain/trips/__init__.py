from .kashmir_7d import KASHMIR_7D_DATA
from .south_india_7d import SOUTH_INDIA_7D_DATA
from .andaman_7d import ANDAMAN_7D_DATA
from .loader import get_trip_data, get_all_trips, TRIP_DATA_REGISTRY

__all__ = [
    "KASHMIR_7D_DATA",
    "SOUTH_INDIA_7D_DATA",
    "ANDAMAN_7D_DATA",
    "get_trip_data",
    "get_all_trips",
    "TRIP_DATA_REGISTRY"
]

# Backward compatibility
SPITI_7D_DATA = KASHMIR_7D_DATA  # Default to Kashmir for backward compat

