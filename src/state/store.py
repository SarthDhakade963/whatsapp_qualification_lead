# State store adapter (Redis/DB)
# For now, using in-memory storage

class StateStore:
    """In-memory state store for development."""
    
    def __init__(self):
        self._store = {}
    
    def get(self, key: str):
        return self._store.get(key)
    
    def set(self, key: str, value: any):
        self._store[key] = value
    
    def delete(self, key: str):
        if key in self._store:
            del self._store[key]

