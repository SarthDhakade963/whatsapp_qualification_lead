import os
import json
from typing import List, Dict, Any, Optional

# Prevent torch from loading if possible (for Windows paging file issues)
# Set environment variables before importing langchain
os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")

from langsmith import traceable

from app.settings import Settings
from llm.prompts import CLASSIFIER_PROMPT, PLANNER_PROMPT, COMPOSER_PROMPT, CATEGORIZER_PROMPT, EXTRACTOR_PROMPT, INTENT_DETECTOR_PROMPT

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


DATE_TERMS = [
    "date", "dates", "start", "end", "departure", "return", "schedule",
    "when does", "available dates", "batch"
]

PACKING_TERMS = [
    "carry", "bring", "pack", "packing", "essentials", "items", "things to"
]

WEATHER_TERMS = [
    "weather", "climate", "temperature", "snow", "rain", "season", "cold", "hot"
]

SAFETY_TERMS = [
    "safe", "safety", "risk", "danger", "secure", "precautions", "concerns"
]

SEAT_TERMS = [
    "seat", "availability", "available", "spots", "vacancy", "vacancies"
]

CATEGORY_TERMS = [
    "recommended", "who is this for", "category", "type of trip", "suitable for"
]



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
            
            # Initialize Flash model for faster classification/categorization tasks
            self.flash_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=api_key,
                temperature=0.0,
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
        self.intent_detector_prompt = ChatPromptTemplate.from_template(INTENT_DETECTOR_PROMPT)
        
        # Initialize parsers
        self.str_parser = StrOutputParser()
        self.json_parser = JsonOutputParser()
    
    def _filter_trip_data(self, question_text: str, trip_data: Dict[str, Any]) -> Dict[str, Any]:
        if not trip_data or not isinstance(trip_data, dict):
            return {}

        question_lower = question_text.lower()

        filtered_data = {}

        # General questions
        if any(phrase in question_lower for phrase in ["tell me about", "what is", "describe", "about"]):
            filtered_data["trip_summary"] = trip_data.get("trip_summary")
            filtered_data["itinerary"] = trip_data.get("itinerary", [])[:7]
            filtered_data["itinerary_highlights"] = trip_data.get("itinerary_highlights", [])

        # Meals
        if any(term in question_lower for term in ["meal", "food", "breakfast", "lunch", "dinner"]):
            filtered_data["inclusions"] = trip_data.get("inclusions", [])
            filtered_data["exclusions"] = trip_data.get("exclusions", [])

        # Safety
        if any(term in question_lower for term in SAFETY_TERMS):
            filtered_data["safety_profile"] = trip_data.get("safety_profile")

        # 🔥 NEW: Dates & Seats
        if any(term in question_lower for term in DATE_TERMS + SEAT_TERMS):
            filtered_data["batches"] = trip_data.get("batches", [])

        # 🔥 NEW: Packing
        if any(term in question_lower for term in PACKING_TERMS):
            filtered_data["things_to_carry"] = trip_data.get("things_to_carry", [])

        # 🔥 NEW: Weather
        if any(term in question_lower for term in WEATHER_TERMS):
            filtered_data["weather_expectation"] = trip_data.get("weather_expectation")

        # 🔥 NEW: Safety
        if any(term in question_lower for term in SAFETY_TERMS):
            filtered_data["safety_profile"] = trip_data.get("safety_profile")

        # 🔥 NEW: Category / Recommended
        if any(term in question_lower for term in CATEGORY_TERMS):
            filtered_data["recommended_for"] = trip_data.get("recommended_for")
            filtered_data["trip_category"] = trip_data.get("trip_category")


        # 🔥 NEW: Category / Recommended
        if any(term in question_lower for term in CATEGORY_TERMS):
            filtered_data["recommended_for"] = trip_data.get("recommended_for")
            filtered_data["trip_category"] = trip_data.get("trip_category")

        # Fallback safety net
        if len(filtered_data) <= 4:
            filtered_data["trip_summary"] = trip_data.get("trip_summary")
            filtered_data["itinerary"] = trip_data.get("itinerary", [])[:7]

        return filtered_data

    
    @traceable(name="classify_question")
    def classify_question(self, question_text: str) -> str:
        """Classify a question into ANSWERABLE, FORBIDDEN, MALFORMED, or HOSTILE."""
        self.call_history.append(("classify", question_text))
        
        try:
            # Use Flash model for faster classification
            chain = self.classifier_prompt | self.flash_llm | self.str_parser
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
    
    @traceable(name="classify_questions_batch")
    def classify_questions_batch(self, questions: List[str]) -> Dict[str, str]:
        """Classify multiple questions in a single LLM call for better performance.
        
        Args:
            questions: List of question texts to classify
            
        Returns:
            Dictionary mapping each question to its classification
        """
        if not questions:
            return {}
        
        # For single question, use existing method
        if len(questions) == 1:
            classification = self.classify_question(questions[0])
            return {questions[0]: classification}
        
        self.call_history.append(("classify_batch", questions))
        
        try:
            # Create batch prompt
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            
            batch_prompt_template = """Classify each question into EXACTLY ONE of these categories: ANSWERABLE, FORBIDDEN, MALFORMED, HOSTILE.

Rules:
- ANSWERABLE: Questions we can answer with our trip information (e.g., pickup details, itinerary, pricing for trips, weather conditions, snowfall expectations)
- FORBIDDEN: Questions about refunds, guarantees about policies/terms, or promises we cannot make (must redirect). Note: Questions about weather/conditions are ANSWERABLE even if they use words like "definitely" - we can answer with available information.
- MALFORMED: Questions that are too short, unclear, or nonsensical (less than 5 characters or no clear meaning)
- HOSTILE: Questions with hostile, offensive, or inappropriate language

Questions:
{questions_text}

Return ONLY a JSON object mapping each question to its classification.
Format: {{"question1": "ANSWERABLE", "question2": "FORBIDDEN"}}
Return ONLY the JSON object, no explanations."""
            
            batch_prompt = ChatPromptTemplate.from_template(batch_prompt_template)
            # Use Flash model for faster batch classification
            chain = batch_prompt | self.flash_llm | self.json_parser
            
            result = chain.invoke({"questions_text": questions_text})
            
            # Parse result - should be a dict mapping questions to classifications
            valid_classes = ["ANSWERABLE", "FORBIDDEN", "MALFORMED", "HOSTILE"]
            question_to_classification = {}
            
            if isinstance(result, dict):
                for q in questions:
                    classification = None
                    
                    # Try exact match first
                    if q in result:
                        classification = result[q]
                    else:
                        # Try partial match (question in key or key in question)
                        for key, value in result.items():
                            if q == key or q in key or key in q:
                                classification = value
                                break
                    
                    # Validate and normalize classification
                    if classification:
                        classification = str(classification).strip().upper()
                        # Extract valid class if it's part of a longer response
                        if classification in valid_classes:
                            question_to_classification[q] = classification
                        else:
                            for valid_class in valid_classes:
                                if valid_class in classification:
                                    question_to_classification[q] = valid_class
                                    break
                            else:
                                # Fallback: use fallback logic for this question
                                question_to_classification[q] = self._classify_fallback(q)
                    else:
                        # Fallback: use fallback logic for this question
                        question_to_classification[q] = self._classify_fallback(q)
                
                return question_to_classification
            
            # Fallback: return classifications using fallback logic
            return {q: self._classify_fallback(q) for q in questions}
                    
        except Exception as e:
            # Fallback on error - try individual calls
            print(f"LLM batch classification error: {e}, falling back to individual calls")
            result = {}
            for q in questions:
                try:
                    result[q] = self.classify_question(q)
                except:
                    result[q] = "ANSWERABLE"
            return result
    
    def _classify_fallback(self, question_text: str) -> str:
        """Fallback classification logic (same as in classify_question)."""
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
            # Use Flash model for faster categorization
            chain = self.categorizer_prompt | self.flash_llm | self.str_parser
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
    
    @traceable(name="categorize_questions_batch")
    def categorize_questions_batch(self, questions: List[str]) -> Dict[str, str]:
        """Categorize multiple questions in a single LLM call for better performance.
        
        Args:
            questions: List of question texts to categorize
            
        Returns:
            Dictionary mapping each question to its category
        """
        if not questions:
            return {}
        
        # For small batches (<=3), individual calls are faster due to less overhead
        if len(questions) <= 3:
            result = {}
            for q in questions:
                result[q] = self.categorize_question(q)
            return result
        
        self.call_history.append(("categorize_batch", questions))
        
        try:
            # Create batch prompt
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            
            batch_prompt_template = """Categorize each question into EXACTLY ONE of these categories: LOGISTICS, COST, ITINERARY, POLICY.

Rules:
- LOGISTICS: Questions about pickup points, transportation, accommodation, hotels, travel arrangements, meeting points, departure/arrival details
- COST: Questions about pricing, costs, fees, payment, budget, expenses, total price, per person cost
- ITINERARY: Questions about schedule, daily activities, what to do each day, places to visit, sightseeing, day-by-day plan, duration
- POLICY: Questions about refund policies, cancellation policies, terms and conditions (though these may be classified as FORBIDDEN earlier)

Questions:
{questions_text}

Return ONLY a JSON object mapping each question to its category.
Format: {{"question1": "LOGISTICS", "question2": "COST"}}
Return ONLY the JSON object, no explanations."""
            
            batch_prompt = ChatPromptTemplate.from_template(batch_prompt_template)
            # Use Flash model for faster batch categorization
            chain = batch_prompt | self.flash_llm | self.json_parser
            
            result = chain.invoke({"questions_text": questions_text})
            
            # Parse result - should be a dict mapping questions to categories
            valid_categories = ["LOGISTICS", "COST", "ITINERARY", "POLICY"]
            question_to_category = {}
            
            if isinstance(result, dict):
                for q in questions:
                    category = None
                    
                    # Try exact match first
                    if q in result:
                        category = result[q]
                    else:
                        # Try partial match (question in key or key in question)
                        for key, value in result.items():
                            if q == key or q in key or key in q:
                                category = value
                                break
                    
                    # Validate and normalize category
                    if category:
                        category = str(category).strip().upper()
                        # Extract valid category if it's part of a longer response
                        if category in valid_categories:
                            question_to_category[q] = category
                        else:
                            for valid_category in valid_categories:
                                if valid_category in category:
                                    question_to_category[q] = valid_category
                                    break
                            else:
                                # Fallback: use fallback logic for this question
                                question_to_category[q] = self._categorize_fallback(q)
                    else:
                        # Fallback: use fallback logic for this question
                        question_to_category[q] = self._categorize_fallback(q)
                
                return question_to_category
            
            # Fallback: return categories using fallback logic
            return {q: self._categorize_fallback(q) for q in questions}
                    
        except Exception as e:
            # Fallback on error - try individual calls
            print(f"LLM batch categorization error: {e}, falling back to individual calls")
            result = {}
            for q in questions:
                try:
                    result[q] = self.categorize_question(q)
                except:
                    result[q] = "LOGISTICS"
            return result
    
    def _categorize_fallback(self, question_text: str) -> str:
        """Fallback categorization logic (same as in categorize_question)."""
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
        
        # Filter trip_data to reduce payload size and improve latency
        filtered_trip_data = self._filter_trip_data(question_text, trip_data)
        
        try:
            # Use Gemini for fact extraction
            chain = self.extractor_prompt | self.llm | self.json_parser
            
            result = chain.invoke({
                "question_text": question_text,
                "trip_data": json.dumps(filtered_trip_data, indent=2)
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
    
    @traceable(name="extract_facts_batch")
    def extract_facts_batch(self, questions: List[str], trip_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract relevant facts for multiple questions in a single LLM call.
        
        Args:
            questions: List of question texts to process
            trip_data: Trip data dictionary
            
        Returns:
            Dictionary mapping each question to its list of facts
        """
        if not questions:
            return {}
        
        if not trip_data or not isinstance(trip_data, dict):
            return {q: [] for q in questions}
        
        # For single question, use existing method
        if len(questions) == 1:
            facts = self.extract_facts(questions[0], trip_data)
            return {questions[0]: facts}
        
        self.call_history.append(("extract_facts_batch", questions))
        
        # Determine what fields are needed based on all questions
        # For batch, include fields needed by any question
        combined_question = " ".join(questions).lower()
        filtered_trip_data = self._filter_trip_data(combined_question, trip_data)
        
        try:
            # Create batch prompt
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            
            batch_prompt_template = """You are a fact extraction system for a travel booking assistant.

Questions:
{questions_text}

Trip Data (JSON):
{trip_data}

Instructions:
- Extract relevant facts from the trip data for EACH question
- Return a JSON object mapping each question to its facts
- Format: {{"question1": ["fact1", "fact2"], "question2": ["fact1"]}}
- Each fact should be a complete, standalone statement
- If a question asks about something not in trip data, return an empty array for that question
- Do not make up information not present in trip data
- Be specific and accurate
- Use natural language for facts (not just raw data values)

Common fields to look for:
- Duration: Check "duration" object (days/nights)
- Accommodation: Check "accommodation" object (stays, room_sharing, type)
- Pickup/Meeting point: Check "logistics" object (meeting_point, pickup)
- Itinerary: Check "itinerary" array
- Pricing: Check "pricing" object
- Meals/Food: Check "inclusions" and "exclusions" arrays, and "itinerary" activities for meal mentions

Return ONLY a JSON object mapping questions to their facts, no explanations."""
            
            batch_prompt = ChatPromptTemplate.from_template(batch_prompt_template)
            chain = batch_prompt | self.llm | self.json_parser
            
            result = chain.invoke({
                "questions_text": questions_text,
                "trip_data": json.dumps(filtered_trip_data, indent=2)
            })
            
            # Parse result - should be a dict mapping questions to facts
            if isinstance(result, dict):
                # Ensure all questions are in the result
                question_to_facts = {}
                result_items = list(result.items())
                
                for idx, q in enumerate(questions):
                    facts = None
                    
                    # Try exact match first
                    if q in result:
                        facts = result[q]
                    else:
                        # Try partial match (question in key or key in question)
                        for key, value in result_items:
                            if q == key or q in key or key in q:
                                facts = value
                                break
                        
                        # If still not found, try by position (index)
                        if facts is None and idx < len(result_items):
                            facts = result_items[idx][1]
                    
                    # Ensure facts is a list of strings
                    if facts is None:
                        facts = []
                    
                    if isinstance(facts, list):
                        question_to_facts[q] = [str(fact) for fact in facts if fact]
                    else:
                        question_to_facts[q] = [str(facts)] if facts else []
                
                return question_to_facts
            
            # Fallback: return empty for all questions
            return {q: [] for q in questions}
                    
        except Exception as e:
            # Fallback on error - try individual calls
            print(f"LLM batch fact extraction error: {e}, falling back to individual calls")
            result = {}
            for q in questions:
                try:
                    result[q] = self.extract_facts(q, trip_data)
                except:
                    result[q] = []
            return result
    
    @traceable(name="compose_answer")
    def compose_answer(self, handler_outputs: List[Dict[str, Any]], normalized_text: str) -> str:
        """Compose final answer from handler outputs using Gemini."""
        self.call_history.append(("compose", handler_outputs))
        
        try:
            # Extract only facts to reduce payload size
            facts_list = []
            for output in handler_outputs:
                facts = output.get("facts", [])
                if isinstance(facts, list):
                    facts_list.extend(facts)
                elif isinstance(facts, str):
                    facts_list.append(facts)
            
            # Use simplified payload - just facts array
            simplified_output = {"facts": facts_list}
            chain = self.composer_prompt | self.flash_llm | self.str_parser
            
            result = chain.invoke({
                "handler_outputs": json.dumps(simplified_output, indent=2),
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
    
    @traceable(name="detect_intent")
    def detect_intent(self, question_text: str) -> str:
        """Detect if question is about SEAT_AVAILABILITY, DATES, or OTHER using LLM."""
        self.call_history.append(("detect_intent", question_text))
        
        try:
            # Use Gemini for intent detection
            chain = self.intent_detector_prompt | self.llm | self.str_parser
            result = chain.invoke({"question_text": question_text})
            
            # Parse and validate result
            intent = result.strip().upper()
            valid_intents = ["SEAT_AVAILABILITY", "DATES", "OTHER"]
            
            # Check if result matches valid intents
            if intent in valid_intents:
                return intent
            
            # Extract intent if it's part of a longer response
            for valid_intent in valid_intents:
                if valid_intent in intent:
                    return valid_intent
                    
        except Exception as e:
            # Fallback on error
            print(f"LLM intent detection error: {e}, using fallback logic")
        
        # Fallback: return OTHER if LLM fails
        return "OTHER"
