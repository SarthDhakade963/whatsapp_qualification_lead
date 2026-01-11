"""Comprehensive tests for Kashmir pickup query: 
'Is pickup included in the Kashmir trip, or do I need to reach Srinagar on my own?'"""

import unittest
import sys
import os

# Add src to path - use absolute path to handle running from different directories
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
_src_dir = os.path.join(_project_root, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Prevent torch loading issues
os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")

# Configure LangSmith BEFORE importing graph modules (IMPORTANT for tracing!)
from app.settings import Settings
settings = Settings()

if settings.langsmith_tracing:
    langsmith_key = settings.effective_langsmith_api_key()
    if langsmith_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_PROJECT"] = settings.effective_langsmith_project()
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        # Handle workspace ID for org-scoped API keys (optional)
        workspace_id = (
            os.getenv("LANGSMITH_WORKSPACE_ID") or 
            settings.langsmith_workspace_id or
            os.getenv("LANGCHAIN_WORKSPACE_ID")
        )
        if workspace_id:
            os.environ["LANGSMITH_WORKSPACE_ID"] = workspace_id
            os.environ["LANGCHAIN_WORKSPACE_ID"] = workspace_id
    elif os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = settings.effective_langsmith_project()
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# NOW import build_graph (after env vars are set)
from graph.state import InputPayload, Questions
from graph.build_graph import build_graph


class TestKashmirPickupQuery(unittest.TestCase):
    """Test suite for Kashmir pickup logistics query."""
    
    TEST_QUERY = "Is pickup included in the Kashmir trip, or do I need to reach Srinagar on my own?"
    EXPECTED_TRIP_ID = "kashmir_zo_trip_TR-4Q7QMQQJ"
    EXPECTED_CATEGORY = "LOGISTICS"
    EXPECTED_CLASSIFICATION = "ANSWERABLE"
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph = build_graph()
        self.initial_state = {
            "input": InputPayload(raw_text=self.TEST_QUERY),
            "questions": Questions()
        }
    
    def test_full_workflow_kashmir_pickup(self):
        """Test full end-to-end workflow for Kashmir pickup query."""
        final_state = self.graph.invoke(self.initial_state.copy())
        
        # Verify workflow completed
        self.assertIn("merged_output", final_state)
        
        # Verify output exists
        merged_output = final_state.get("merged_output")
        self.assertIsNotNone(merged_output, "Should have merged output")
        
        if merged_output:
            final_text = merged_output.get("final_text", "")
            self.assertGreater(len(final_text), 0, "Should generate a response")
            
            # Output should mention pickup/Srinagar (case-insensitive)
            text_lower = final_text.lower()
            self.assertTrue(
                "pickup" in text_lower or "srinagar" in text_lower or "meeting" in text_lower,
                f"Response should mention pickup/Srinagar/meeting point. Got: {final_text[:200]}"
            )
            
            print(f"\n[OK] Workflow completed successfully")
            print(f"Response: {final_text}")
        
        # Verify trip context was resolved
        answerable_processing = final_state.get("answerable_processing")
        if answerable_processing:
            trip_context = answerable_processing.get("trip_context")
            if trip_context:
                trip_id = trip_context.get("trip_id", "")
                confidence = trip_context.get("confidence", "")
                
                print(f"\n[OK] Trip context resolved:")
                print(f"  Trip ID: {trip_id}")
                print(f"  Confidence: {confidence}")
                
                # Verify it resolved to Kashmir trip
                if trip_id:
                    self.assertEqual(trip_id, self.EXPECTED_TRIP_ID,
                                   f"Should resolve to Kashmir trip, got {trip_id}")
                else:
                    print(f"  Warning: Trip ID is empty - trip context not fully resolved")
    
    def test_query_variations(self):
        """Test variations of the Kashmir pickup query."""
        variations = [
            "Is pickup included in Kashmir trip?",
            "Do I need to reach Srinagar myself for Kashmir trip?",
            "Kashmir trip pickup location?",
            "Where is the meeting point for Kashmir trip?",
            "Kashmir trip - pickup or self travel to Srinagar?",
        ]
        
        for query in variations:
            with self.subTest(query=query):
                initial_state = {
                    "input": InputPayload(raw_text=query),
                    "questions": Questions()
                }
                
                final_state = self.graph.invoke(initial_state)
                
                # Verify workflow completed
                merged_output = final_state.get("merged_output")
                if merged_output:
                    final_text = merged_output.get("final_text", "")
                    self.assertGreater(len(final_text), 0,
                                     f"Should generate response for: {query}")
                    
                    # Should be relevant to pickup/logistics
                    text_lower = final_text.lower()
                    relevant_terms = ["pickup", "srinagar", "meeting", "arrival", "logistics"]
                    self.assertTrue(
                        any(term in text_lower for term in relevant_terms),
                        f"Response should be relevant to pickup/logistics for query: {query}"
                    )
                    print(f"\n[OK] Query variation handled: {query[:50]}...")
    
    def test_expected_response_content(self):
        """Test that response contains expected information about Kashmir pickup."""
        final_state = self.graph.invoke(self.initial_state.copy())
        merged_output = final_state.get("merged_output")
        
        if merged_output:
            final_text = merged_output.get("final_text", "").lower()
            
            # Based on Kashmir trip data:
            # - pickup_not_included: "true"
            # - meeting_point: "Srinagar (arrival point)"
            
            # Response should indicate one of:
            # 1. Pickup is not included / You need to reach Srinagar on your own
            # 2. Meeting point is Srinagar
            # 3. Pickup not provided, meeting point is Srinagar
            
            expected_indicators = [
                "pickup" in final_text and ("not" in final_text or "included" not in final_text or "srinagar" in final_text),
                "meeting point" in final_text and "srinagar" in final_text,
                "srinagar" in final_text and ("arrival" in final_text or "meeting" in final_text),
            ]
            
            self.assertTrue(
                any(expected_indicators),
                f"Response should indicate pickup not included or meeting point is Srinagar. "
                f"Got: {merged_output.get('final_text', '')}"
            )
            
            print(f"\n[OK] Response contains expected pickup information")


class TestKashmirPickupDataStructure(unittest.TestCase):
    """Test Kashmir trip data structure for pickup information."""
    
    def test_kashmir_trip_pickup_data(self):
        """Test that Kashmir trip data has correct pickup structure."""
        from domain.trips.loader import get_trip_data
        
        trip_id = "kashmir_zo_trip_TR-4Q7QMQQJ"
        trip_data = get_trip_data(trip_id)
        
        self.assertIsNotNone(trip_data, "Should be able to load Kashmir trip data")
        self.assertEqual(trip_data["trip_id"], trip_id)
        
        # Check logistics structure
        logistics = trip_data.get("logistics", {})
        self.assertIn("pickup_not_included", logistics or "pickup" in logistics)
        self.assertIn("meeting_point", logistics or "meeting" in logistics)
        self.assertIn("srinagar", str(logistics).lower())
        
        print(f"\n[OK] Kashmir trip data structure is correct")
    
    def test_logistics_handler_extracts_pickup_info(self):
        """Test that logistics handler can extract pickup info from Kashmir data."""
        from domain.trips.loader import get_trip_data
        
        trip_id = "kashmir_zo_trip_TR-4Q7QMQQJ"
        trip_data = get_trip_data(trip_id)
        
        # Verify data structure is correct
        self.assertIn("logistics", trip_data)
        logistics = trip_data["logistics"]
        self.assertIn("pickup_not_included", logistics)
        self.assertIn("meeting_point", logistics)
        self.assertEqual(logistics["pickup_not_included"], "true")
        self.assertIn("Srinagar", logistics["meeting_point"])
        
        # Check pickup details if available
        pickup = logistics.get("pickup", {})
        if pickup:
            self.assertIn("included", pickup or "details" in pickup)
            print(f"\n[OK] Pickup details available in trip data")


if __name__ == "__main__":
    unittest.main(verbosity=2)
