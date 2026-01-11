from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from .env that we don't define
        env_ignore_empty=True
    )
    
    app_name: str = "WhatsApp Lead Qualification"
    debug: bool = False
    
    # Gemini API - will read from env vars automatically (case-insensitive)
    gemini_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # LangSmith tracing - will read from env vars automatically
    langsmith_api_key: Optional[str] = None
    langchain_api_key: Optional[str] = None
    langsmith_project: Optional[str] = None
    langchain_project: Optional[str] = None
    langsmith_workspace_id: Optional[str] = None
    langsmith_tracing: bool = True
    
    # Model configuration
    gemini_model: str = "gemini-2.5-pro"
    gemini_video_model: Optional[str] = None
    
    def effective_gemini_api_key(self) -> str:
        """Get Gemini API key from any available source."""
        return (
            self.gemini_api_key or 
            self.google_api_key or 
            os.getenv("GEMINI_API_KEY", "") or 
            os.getenv("GOOGLE_API_KEY", "")
        )
    
    def effective_langsmith_api_key(self) -> str:
        """Get LangSmith API key from any available source."""
        return (
            self.langsmith_api_key or 
            self.langchain_api_key or 
            os.getenv("LANGSMITH_API_KEY", "") or 
            os.getenv("LANGCHAIN_API_KEY", "")
        )
    
    def effective_langsmith_project(self) -> str:
        """Get LangSmith project from any available source."""
        return (
            self.langsmith_project or 
            self.langchain_project or 
            os.getenv("LANGSMITH_PROJECT", "") or 
            os.getenv("LANGCHAIN_PROJECT", "whatsapp-lead-qualification")
        )

