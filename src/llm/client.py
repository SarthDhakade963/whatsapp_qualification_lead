import os
import json
from typing import List, Dict, Any, Optional

# Prevent torch from loading if possible (for Windows paging file issues)
# Set environment variables before importing langchain
os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")

from langsmith import traceable

from app.settings import Settings
from llm.prompts import CLASSIFIER_PROMPT, PLANNER_PROMPT, COMPOSER_PROMPT, CATEGORIZER_PROMPT, EXTRACTOR_PROMPT

# Lazy imports to avoid loading torch/transformers if not needed
# Catch all exceptions including OSError from torch DLL loading on Windows
LANGCHAIN_AVAILABLE = False
ChatGoogleGenerativeAI = None
ChatPromptTemplate = None
StrOutputParser = None
JsonOutputParser = None
_import_error = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    LANGCHAIN_AVAILABLE = True
except Exception as e:
    # Handle import errors gracefully (e.g., torch loading issues, OSError on Windows)
    _import_error = e
    LANGCHAIN_AVAILABLE = False


class LLMClient:
    """Gemini 2.5 Pro LLM client with LangSmith tracing for classification, planning, and composition."""
    
    def __init__(self):
        if not LANGCHAIN_AVAILABLE:
            error_msg = "LangChain dependencies not available."
            if _import_error:
                error_msg += f" Error: {_import_error}"
                if isinstance(_import_error, OSError) and "torch" in str(_import_error).lower():
                    error_msg += "\n\nThis appears to be a Windows paging file issue with PyTorch. "
                    error_msg += "Try one of these solutions:\n"
                    error_msg += "1. Increase your Windows paging file size\n"
                    error_msg += "2. Install torch CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                    error_msg += "3. Set environment variable: set TRANSFORMERS_NO_TORCH=1"
            raise ImportError(error_msg)
        
        settings = Settings()
        
        # Initialize Gemini 2.5 Pro (or fallback to available model)
        model_name = settings.gemini_model or settings.gemini_video_model or "gemini-2.0-flash-exp"
        
        # Check if API key is available (try multiple sources)
        api_key = settings.effective_gemini_api_key()
        
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY "
                "in .env file or environment variables."
            )
        
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.0,  # Deterministic for classification
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Gemini LLM: {e}. "
                "Please check your API key and model name."
            )
        
        self.call_history = []
        
        # Initialize prompt templates
        self.classifier_prompt = ChatPromptTemplate.from_template(CLASSIFIER_PROMPT)
        self.planner_prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
        self.composer_prompt = ChatPromptTemplate.from_template(COMPOSER_PROMPT)
        self.categorizer_prompt = ChatPromptTemplate.from_template(CATEGORIZER_PROMPT)
        self.extractor_prompt = ChatPromptTemplate.from_template(EXTRACTOR_PROMPT)
        
        # Initialize parsers
        self.str_parser = StrOutputParser()
        self.json_parser = JsonOutputParser()
    
    @traceable(name="classify_question")
    def classify_question(self, question_text: str) -> str:
        """Classify a question into ANSWERABLE, FORBIDDEN, MALFORMED, or HOSTILE."""
        self.call_history.append(("classify", question_text))
        
        try:
            # Use Gemini for classification
            chain = self.classifier_prompt | self.llm | self.str_parser
            result = chain.invoke({"question_text": question_text})
            
            # Parse and validate result
            classification = result.strip().upper()
            valid_classes = ["ANSWERABLE", "FORBIDDEN", "MALFORMED", "HOSTILE"]
            
            # Check if result matches valid classes
            if classification in valid_classes:
                return classification
            
            # Extract classification if it's part of a longer response
            for valid_class in valid_classes:
                if valid_class in classification:
                    return valid_class
                    
        except Exception as e:
            # Fallback on error
            print(f"LLM classification error: {e}, using fallback logic")
        
        # Fallback logic if LLM doesn't return exact match
        text_lower = question_text.lower()
        if "refund" in text_lower or "guarantee" in text_lower:
            return "FORBIDDEN"
        elif len(question_text.strip()) < 5:
            return "MALFORMED"
        elif any(word in text_lower for word in ["stupid", "hate", "terrible"]):
            return "HOSTILE"
        else:
            return "ANSWERABLE"
    
    @traceable(name="categorize_question")
    def categorize_question(self, question_text: str) -> str:
        """Categorize an answerable question into LOGISTICS, COST, ITINERARY, or POLICY using LLM."""
        self.call_history.append(("categorize", question_text))
        
        try:
            # Use Gemini for categorization
            chain = self.categorizer_prompt | self.llm | self.str_parser
            result = chain.invoke({"question_text": question_text})
            
            # Parse and validate result
            category = result.strip().upper()
            valid_categories = ["LOGISTICS", "COST", "ITINERARY", "POLICY"]
            
            # Check if result matches valid categories
            if category in valid_categories:
                return category
            
            # Extract category if it's part of a longer response
            for valid_category in valid_categories:
                if valid_category in category:
                    return valid_category
                    
        except Exception as e:
            # Fallback on error
            print(f"LLM categorization error: {e}, using fallback logic")
        
        # Fallback logic if LLM doesn't return exact match
        text_lower = question_text.lower()
        
        if "pickup" in text_lower or "transport" in text_lower or "accommodation" in text_lower or "hotel" in text_lower or "meeting point" in text_lower:
            return "LOGISTICS"
        elif "cost" in text_lower or "price" in text_lower or "pricing" in text_lower or "payment" in text_lower or "budget" in text_lower:
            return "COST"
        elif "itinerary" in text_lower or "day" in text_lower or "schedule" in text_lower or "activities" in text_lower or "places to visit" in text_lower:
            return "ITINERARY"
        elif "policy" in text_lower or "refund" in text_lower or "cancellation" in text_lower:
            return "POLICY"
        else:
            return "LOGISTICS"
    
    @traceable(name="plan_answer")
    def plan_answer(self, structured_questions: List[Dict[str, Any]], trip_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an answer plan using Gemini."""
        self.call_history.append(("plan", structured_questions))
        
        try:
            # Use Gemini for planning
            chain = self.planner_prompt | self.llm | self.json_parser
            
            result = chain.invoke({
                "structured_questions": json.dumps(structured_questions, indent=2),
                "trip_context": json.dumps(trip_context, indent=2)
            })
            
            # Validate and ensure proper structure
            if isinstance(result, dict) and "answer_blocks" in result:
                blocks = result["answer_blocks"]
                
                # Create blocks with proper structure
                from utils.ids import generate_block_id
                
                formatted_blocks = []
                for block in blocks:
                    if isinstance(block, dict):
                        category = block.get("category") or ""
                        question_ids = block.get("question_ids", [])
                        
                        # If question_ids is not a list, try to get from structured_questions
                        if not isinstance(question_ids, list):
                            # Try to match by category
                            question_ids = [q["id"] for q in structured_questions if q.get("category") == category]
                        
                        handler_map = {
                            "LOGISTICS": "logistics_handler",
                            "COST": "pricing_handler",
                            "ITINERARY": "itinerary_handler",
                            "POLICY": "pricing_handler"
                        }
                        
                        # Determine handler from category if not provided
                        handler = block.get("handler") or handler_map.get(category, "logistics_handler")
                        
                        formatted_blocks.append({
                            "block_id": block.get("block_id") or generate_block_id(),
                            "question_ids": question_ids if isinstance(question_ids, list) else [question_ids] if question_ids else [],
                            "handler": handler,
                            "answer_style": block.get("answer_style", "HIGH_LEVEL" if len(question_ids) == 1 else "DETAILED")
                        })
                
                if formatted_blocks:
                    return {"answer_blocks": formatted_blocks}
                    
        except Exception as e:
            # Fallback on error
            print(f"LLM planning error: {e}, using fallback logic")
        
        # Fallback: group by category
        blocks = []
        category_groups: Dict[str, List[str]] = {}
        
        for q in structured_questions:
            cat = q.get("category", "LOGISTICS")
            q_id = q.get("id", "")
            if cat not in category_groups:
                category_groups[cat] = []
            if q_id:
                category_groups[cat].append(q_id)
        
        from utils.ids import generate_block_id
        
        for category, question_ids in category_groups.items():
            if not question_ids:
                continue
                
            handler_map = {
                "LOGISTICS": "logistics_handler",
                "COST": "pricing_handler",
                "ITINERARY": "itinerary_handler",
                "POLICY": "pricing_handler"
            }
            
            blocks.append({
                "block_id": generate_block_id(),
                "question_ids": question_ids,
                "handler": handler_map.get(category, "logistics_handler"),
                "answer_style": "HIGH_LEVEL" if len(question_ids) == 1 else "DETAILED"
            })
        
        return {"answer_blocks": blocks}
    
    @traceable(name="extract_facts")
    def extract_facts(self, question_text: str, trip_data: Dict[str, Any]) -> List[str]:
        """Extract relevant facts from trip data using LLM."""
        self.call_history.append(("extract_facts", question_text))
        
        if not trip_data or not isinstance(trip_data, dict):
            return []
        
        try:
            # Use Gemini for fact extraction
            chain = self.extractor_prompt | self.llm | self.json_parser
            
            result = chain.invoke({
                "question_text": question_text,
                "trip_data": json.dumps(trip_data, indent=2)
            })
            
            # Parse result - should be a list of strings
            if isinstance(result, list):
                return [str(fact) for fact in result if fact]
            elif isinstance(result, dict):
                # Handle case where LLM returns {"facts": [...]}
                facts = result.get("facts", [])
                if isinstance(facts, list):
                    return [str(fact) for fact in facts if fact]
            
            return []
                    
        except Exception as e:
            # Fallback on error
            print(f"LLM fact extraction error: {e}, using fallback logic")
            return []
    
    @traceable(name="compose_answer")
    def compose_answer(self, handler_outputs: List[Dict[str, Any]], normalized_text: str) -> str:
        """Compose final answer from handler outputs using Gemini."""
        self.call_history.append(("compose", handler_outputs))
        
        try:
            # Use Gemini for composition
            chain = self.composer_prompt | self.llm | self.str_parser
            
            result = chain.invoke({
                "handler_outputs": json.dumps(handler_outputs, indent=2),
                "normalized_text": normalized_text
            })
            
            # Validate result
            if result and result.strip():
                return result.strip()
                
        except Exception as e:
            # Fallback on error
            print(f"LLM composition error: {e}, using fallback logic")
        
        # Fallback: combine facts directly
        answer_parts = []
        for output in handler_outputs:
            facts = output.get("facts", [])
            if isinstance(facts, list):
                answer_parts.extend(facts)
            elif isinstance(facts, str):
                answer_parts.append(facts)
        
        return " ".join(answer_parts) if answer_parts else "I'm here to help. Could you provide more details about your question?"
