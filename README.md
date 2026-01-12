# WhatsApp Lead Qualification System

An intelligent conversational AI system built with LangGraph for automating WhatsApp lead qualification and customer support for travel bookings. The system uses Google Gemini AI to understand user queries, extract trip information, and provide empathetic, context-aware responses.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Behaviors System](#behaviors-system)
- [Testing](#testing)
- [Workflow](#workflow)

## 🎯 Overview

This system automates the initial stages of customer interaction for a travel booking service. It:

- **Understands** natural language queries about trips, pricing, logistics, and policies
- **Categorizes** questions into ANSWERABLE, FORBIDDEN, MALFORMED, or HOSTILE
- **Resolves** trip context automatically from user messages
- **Provides** empathetic, context-aware responses
- **Escalates** complex or sensitive queries to human agents
- **Tracks** conversation state and history for context

## ✨ Features

### Core Capabilities

1. **Intelligent Question Classification**

   - Automatically categorizes questions using LLM-based classification
   - Handles malformed, hostile, and forbidden questions appropriately

2. **Dynamic Trip Discovery**

   - Automatically discovers and loads trip data from `src/domain/trips/`
   - No manual registration required - just add trip files

3. **Context-Aware Responses**

   - Resolves trip context from keywords in user messages
   - Provides relevant information based on trip data

4. **Centralized Behaviors System**

   - **Empathetic Responses**: Handles group size, gender ratio, decision confirmation queries
   - **Seat Availability**: Checks seat availability and suggests next available dates
   - **Call Request Handling**: Multi-turn flow for call requests with availability time collection
   - **Booking Confirmation**: Celebrates bookings with "Zo Zo 😍"

5. **Policy Management**

   - Refund/cancellation policy handling
   - Discount/offer boundary messages
   - Guarantee policy responses

6. **Conversation Memory**

   - Maintains conversation history
   - Uses hybrid approach: recent messages + structured state

7. **Escalation Management**
   - Automatically escalates forbidden questions
   - Handles call requests with conversation summary
   - Tracks escalation flags and risk levels

## 🏗️ Architecture

The system is built using **LangGraph**, a framework for building stateful, multi-step agentic workflows.

### Workflow Graph

```
Entry → Pipeline → Non-Skippable → Post-Processing → End
         ↓              ↓
    Classification   Handlers
    Partitioning     (Itinerary, Pricing, Logistics)
    Merging
```

### State Management

The system uses a `ConversationWorkflowState` TypedDict that includes:

- `input`: Raw user message
- `questions`: Classified and partitioned questions
- `answerable_processing`: Trip context and extracted facts
- `skippable_actions`: Boundaries and clarifications
- `merged_output`: Final response text
- `interaction_state`: Decision stage and escalation flags
- `conversation_history`: Recent message history

## 📦 Installation

### Prerequisites

- Python 3.10+
- Google Gemini API key
- (Optional) LangSmith API key for tracing

### Getting API Keys

**Google Gemini API Key (Required):**

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey) or [Google Cloud Console](https://console.cloud.google.com/)
2. Sign in with your Google account
3. Navigate to API Keys section
4. Click "Create API Key" or "Get API Key"
5. Copy the generated key

**LangSmith API Key (Optional, for tracing):**

1. Visit [LangSmith](https://smith.langchain.com/)
2. Sign up or log in
3. Go to Settings → API Keys
4. Create a new API key
5. Copy the key and workspace ID (found in your account settings)

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd "Whatsapp Lead Qualification"
   ```

2. **Create and activate virtual environment**

   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   # On Windows (CMD)
   venv\Scripts\activate.bat
   # On Linux/Mac
   source venv/bin/activate
   ```

   **Note:** After activation, you should see `(venv)` in your terminal prompt.

3. **Install dependencies**

   ```bash
   python -m pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Copy the example environment file and fill in your API keys:

   ```bash
   # On Windows (PowerShell)
   Copy-Item .env.example .env

   # On Linux/Mac
   cp .env.example .env
   ```

   Then edit the `.env` file and fill in the appropriate values:

   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   GOOGLE_API_KEY=your_gemini_api_key_here  # Alternative name

   # Optional: LangSmith tracing
   LANGSMITH_API_KEY=your_langsmith_key
   LANGSMITH_PROJECT=whatsapp-lead-qualification
   LANGSMITH_WORKSPACE_ID=your_workspace_id
   ```

   **Note:** The `.env` file is already in `.gitignore` and will not be committed to the repository.

5. **Run the application**

   Make sure your virtual environment is activated, then you can run the application using any of the methods described in the [Usage](#-usage) section below.

## ⚙️ Configuration

Configuration is managed through `src/app/settings.py` using Pydantic Settings:

- **API Keys**: Automatically loaded from environment variables
- **Model Selection**: Defaults to `gemini-2.5-pro`
- **Tracing**: LangSmith tracing enabled by default (can be disabled)

## 🚀 Usage

After completing the installation and configuration steps above, you can run the application:

### Streamlit Chat Interface (Recommended for Testing)

```bash
streamlit run streamlit_chat.py
```

This provides an interactive chat interface where you can:

- Test conversation flows
- View response metadata
- See conversation history
- Test various query types

### Command Line Testing

```bash
python run_test.py
```

### Programmatic Usage

```python
from graph.build_graph import build_graph
from graph.state import InputPayload, Questions

# Build the graph
graph = build_graph()

# Create initial state
initial_state = {
    "input": InputPayload(raw_text="Tell me about the Spiti trip?"),
    "questions": Questions()
}

# Run the workflow
final_state = graph.invoke(initial_state)

# Get response
response = final_state["merged_output"]["final_text"]
print(response)
```

## 📁 Project Structure

```
Whatsapp Lead Qualification/
├── src/
│   ├── app/                    # Application settings and FastAPI entry
│   │   ├── main.py            # FastAPI app (placeholder)
│   │   └── settings.py         # Configuration management
│   │
│   ├── domain/                 # Domain logic and data
│   │   ├── behaviors/         # Centralized behaviors
│   │   │   ├── empathetic_responses.py
│   │   │   └── seat_availability.py
│   │   ├── policies/          # Policy definitions
│   │   │   ├── refunds.py
│   │   │   ├── discounts.py
│   │   │   └── guarantees.py
│   │   └── trips/             # Trip data files
│   │       ├── loader.py      # Auto-discovery of trips
│   │       ├── kashmir_7d.py
│   │       ├── spiti_7d.py
│   │       └── ...
│   │
│   ├── graph/                  # LangGraph workflow
│   │   ├── build_graph.py     # Graph construction
│   │   ├── state.py           # State definitions
│   │   └── nodes/             # Workflow nodes
│   │       ├── entry/         # Entry point
│   │       ├── pipeline/      # Question processing
│   │       ├── non_skippable/ # Answerable question handlers
│   │       ├── skippable/     # Boundary handlers
│   │       └── post_processing/
│   │
│   ├── llm/                    # LLM client and prompts
│   │   ├── client.py          # Gemini client wrapper
│   │   └── prompts/           # Prompt templates
│   │
│   ├── state/                  # State management
│   │   ├── store.py           # StateStore backend
│   │   └── memory.py          # ConversationMemory helper
│   │
│   └── utils/                  # Utility functions
│       ├── behaviors.py       # Behavior checkers
│       ├── state_adapter.py  # State conversion helpers
│       └── ...
│
├── tests/                      # Test files
├── streamlit_chat.py          # Streamlit UI
├── run_test.py                # CLI test runner
└── requirements.txt           # Dependencies
```

## 🔧 Key Components

### 1. Graph Nodes

#### Entry

- **`inbound_message`**: Processes incoming messages

#### Pipeline

- **`normalize_and_split`**: Normalizes and splits messages into atomic questions
- **`classify_each_question`**: Classifies each question using LLM
- **`partition_questions`**: Separates answerable from skippable questions
- **`merge_outputs`**: Merges all outputs into final response

#### Non-Skippable (Answerable Questions)

- **`normalize_and_structure`**: Structures questions for processing
- **`resolve_trip_context`**: Identifies relevant trip from keywords
- **`answer_planner`**: Plans answer structure
- **`handlers/`**: Domain-specific handlers
  - `itinerary.py`: Handles itinerary questions
  - `pricing.py`: Handles pricing and policy questions
  - `logistics.py`: Handles logistics questions
- **`merge_handler_outputs`**: Combines handler outputs
- **`compose_answer`**: Composes final answer text

#### Skippable (Boundary Questions)

- **`malformed`**: Handles unclear questions
- **`forbidden`**: Handles questions requiring human intervention
- **`hostile`**: Handles hostile/abusive messages

#### Post-Processing

- **`update_interaction_state`**: Updates decision stage and escalation flags
- **`post_answer_action`**: Performs post-answer actions

### 2. LLM Client

The `LLMClient` class (`src/llm/client.py`) provides:

- Question classification
- Fact extraction from trip data
- Answer planning
- Answer composition

Uses Google Gemini models via LangChain.

### 3. Trip Data System

Trips are automatically discovered from `src/domain/trips/`:

- Each trip file exports a `*_DATA` dictionary
- Must include `trip_id` field
- Loaded automatically via `loader.py`

Example trip structure:

```python
KASHMIR_7D_DATA = {
    "trip_id": "kashmir_zo_trip_TR-4Q7QMQQJ",
    "destination": "Kashmir Valley",
    "duration": "7 days",
    "description": "...",
    "batches": [...],
    # ... more fields
}
```

## 🎭 Behaviors System

The behaviors system provides centralized, rule-based responses for specific query patterns.

### Empathetic Responses

Located in `src/domain/behaviors/empathetic_responses.py`:

- **Group Size Queries**: "Group trips like this usually have around 8–12 participants..."
- **Gender Ratio Queries**: "Usually, we have around 3–4 women travelers..."
- **Decision Confirmation**: "No problem! Take your time..."

### Seat Availability

Located in `src/domain/behaviors/seat_availability.py`:

- Checks trip batches for seat availability
- Extracts dates from user queries
- Provides specific responses:
  - "Limited seats available — book soon to secure your spot."
  - "Unfortunately, we do not have seats on this date, but we do have seats on [next date]."

### Call Request Handling

Located in `src/utils/behaviors.py`:

**First Request:**

```
User: "Can we get on a quick call"
Bot: "I can help with that 🙂
      Before I arrange a call, could you briefly share:
      1. What you'd like to discuss on the call?
      2. Your preferred time/availability for the call?"
```

**Follow-up:**

```
User: "I want to book the trip and a little quick to not miss the remaining two slots... Call me now"
Bot: "Perfect! I've noted your request.

      Our team will reach out to you at your preferred time."
```

The system:

- Extracts discussion points from the user's response
- Extracts availability time
- Generates a summary for the human agent
- Escalates with `escalation_flag: True`

### Booking Confirmation

Detects booking confirmations and responds with "Zo Zo 😍". If concerns are also present, processes them normally.

## 🔄 Workflow

### Question Processing Flow

1. **Input Processing**

   - User message received
   - Normalized and split into atomic questions

2. **Classification**

   - Each question classified as ANSWERABLE, FORBIDDEN, MALFORMED, or HOSTILE

3. **Partitioning**

   - Questions split into skippable and non-skippable

4. **Answerable Path**

   - Trip context resolved
   - Appropriate handler selected (itinerary/pricing/logistics)
   - Facts extracted using LLM
   - Behaviors checked (empathetic, seat availability, etc.)
   - Answer composed

5. **Skippable Path**

   - Boundary messages added for forbidden questions
   - Escalation flags set

6. **Merging**

   - All outputs merged
   - Special behaviors checked (call requests, booking confirmations)
   - Final response generated

7. **Post-Processing**
   - Interaction state updated
   - Escalation flags finalized

## 🧪 Testing

### Streamlit Interface

```bash
streamlit run streamlit_chat.py
```

### Unit Tests

```bash
python tests/test_kashmir_response.py
python tests/test_kashmir_pickup_query.py
```

### Example Test Scenarios

1. **Trip Information**

   - "Tell me about the Spiti trip?"
   - "What's included in the Kashmir package?"

2. **Pricing & Policies**

   - "What's the price for Spiti?"
   - "What's your cancellation policy?"
   - "Any discounts for first-time travelers?"

3. **Logistics**

   - "Is pickup included?"
   - "How do I reach the starting point?"

4. **Behaviors**

   - "How many people have registered?"
   - "Are seats available on 24th January?"
   - "Can we get on a quick call"

5. **Booking**
   - "I just booked the trip!"
   - "Booked... but after that no update"

## 📝 Adding New Trips

1. Create a new file in `src/domain/trips/` (e.g., `goa_5d.py`)
2. Define trip data with `*_DATA` constant:
   ```python
   GOA_5D_DATA = {
       "trip_id": "goa_zo_trip_TR-XXXXX",
       "destination": "Goa",
       "duration": "5 days",
       # ... other fields
   }
   ```
3. The system will automatically discover and register it!

## 🎯 Key Behaviors Reference

### Booking Confirmation

- Keywords: "booked", "just booked", "booking is confirmed", "payment is done"
- Response: "Zo Zo 😍"
- Note: If concerns are present, processes them normally

### Discount/Offer Questions

- Keywords: "discount", "offer", "deal", "promo", "coupon", "cheaper"
- Response: "Aaah, these are the best price we are offering 🤩"

### Refund/Cancellation

- Informational: Provides full policy text
- Transactional: Boundary message for guarantees

### Call Requests

- First: Asks for discussion points and availability
- Follow-up: Escalates with summary

## 🔍 Debugging

### LangSmith Tracing

If enabled, all LLM calls are traced in LangSmith:

- View prompts and responses
- Analyze token usage
- Debug classification issues

### Response Metadata

The Streamlit interface shows:

- Trip ID and confidence
- Decision stage
- Escalation status
- Conversation history

## 📚 Dependencies

- **LangGraph**: Workflow orchestration
- **Google Gemini**: LLM for classification and fact extraction
- **LangChain**: LLM integration
- **Pydantic**: Data validation
- **Streamlit**: Testing interface
- **LangSmith**: (Optional) Tracing and monitoring

## 🤝 Contributing

When adding new features:

1. **New Behaviors**: Add to `src/domain/behaviors/` and update `src/utils/behaviors.py`
2. **New Handlers**: Add to `src/graph/nodes/non_skippable/handlers/`
3. **New Policies**: Add to `src/domain/policies/`
4. **New Trips**: Add files to `src/domain/trips/` (auto-discovered)

## 🔧 Troubleshooting

### Common Issues

**"ModuleNotFoundError" or import errors:**

- Make sure your virtual environment is activated (you should see `(venv)` in your terminal prompt)
- Verify all dependencies are installed: `python -m pip install -r requirements.txt`
- Check that you're running commands from the project root directory
- Ensure Python 3.10+ is being used: `python --version`

**Virtual environment activation issues:**

- On Windows PowerShell, if you get an execution policy error, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Make sure you're in the project root directory when activating
- Verify Python is installed correctly: `python --version`
- On Windows, you can also use: `venv\Scripts\activate.bat` (CMD) or `.\venv\Scripts\Activate.ps1` (PowerShell)

**"API key not found" or authentication errors:**

- Verify your `.env` file exists in the project root
- Check that `GEMINI_API_KEY` is set correctly (no quotes, no extra spaces)
- Ensure the API key is valid and not expired
- Try restarting your terminal/IDE after creating `.env`

**Streamlit not starting:**

- Make sure your virtual environment is activated
- Verify Streamlit is installed: `pip install streamlit`
- Try running: `python -m streamlit run streamlit_chat.py`
- Check if port 8501 is already in use (try a different port: `streamlit run streamlit_chat.py --server.port 8502`)

**LangSmith tracing not working:**

- Verify `LANGSMITH_API_KEY` is set in `.env` (optional - not required)
- Check that `LANGCHAIN_TRACING_V2=true` is set if using tracing
- Ensure your API key has the correct permissions

**Trip data not loading:**

- Verify trip files are in `src/domain/trips/`
- Check that trip files export `*_DATA` dictionary with `trip_id` field
- Ensure trip files are valid Python syntax (no syntax errors)
- Check console for any import errors

**"No space left on device" errors:**

- Clean up Python cache: Remove `__pycache__` directories
- Free up disk space on your system
- Clean git objects if using git: `git gc --prune=now`

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

Built with:

- LangGraph for workflow management
- Google Gemini for AI capabilities
- LangSmith for observability

---

For questions or issues, please refer to the code documentation or create an issue in the repository.
