"""Trip data loader - automatically discovers and loads trip data by trip_id."""

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Optional


def _discover_trip_data() -> Dict[str, Dict]:
    """Automatically discover and load all trip data files."""
    registry = {}
    package_path = Path(__file__).parent
    package_name = __name__.rsplit('.', 1)[0] if '.' in __name__ else 'domain.trips'
    
    # Scan for Python files in the trips directory
    for module_info in pkgutil.iter_modules([str(package_path)]):
        module_name = module_info.name
        # Skip non-trip files (loader, __init__, __pycache__)
        if module_name in ['loader', '__init__'] or module_name.startswith('__'):
            continue
        
        try:
            # Import the module
            module = importlib.import_module(f'.{module_name}', package=package_name)
            # Look for *_DATA constant (e.g., KASHMIR_7D_DATA, SPITI_7D_DATA)
            for attr_name in dir(module):
                if attr_name.endswith('_DATA') and not attr_name.startswith('_'):
                    trip_data = getattr(module, attr_name)
                    if isinstance(trip_data, dict) and 'trip_id' in trip_data:
                        trip_id = trip_data['trip_id']
                        registry[trip_id] = trip_data
        except Exception:
            # Silently skip modules that can't be imported
            continue
    
    return registry


# Auto-discover and register all trips
TRIP_DATA_REGISTRY = _discover_trip_data()


def get_trip_data(trip_id: str) -> Optional[Dict]:
    """Get trip data by trip_id."""
    return TRIP_DATA_REGISTRY.get(trip_id)


def get_all_trips() -> Dict[str, Dict]:
    """Get all available trips."""
    return TRIP_DATA_REGISTRY.copy()
