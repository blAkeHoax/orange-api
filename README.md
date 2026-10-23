# Orange API

Orange API is the AI inference and model orchestration service for the Orange AI ecosystem. Built with FastAPI, it provides high-performance, structured inference endpoints and intelligent capability-based model routing powered by NVIDIA's OpenAI-compatible model platform.

---

## Features & Architecture

### 1. NVIDIA Model Provider Foundation
Orange API communicates with NVIDIA's OpenAI-compatible API (`https://integrate.api.nvidia.com/v1`) using direct, asynchronous-ready HTTP clients (`httpx`). It includes typed error mapping, zero fallback on provider failures, and complete isolation of credentials from client applications.

### 2. Four-Model Foundation
The foundation consists of four specialized models mapped by internal keys:
- **`gpt_oss_20b`** (`openai/gpt-oss-20b`): High-speed conversational reasoning and general text tasks.
- **`nemotron_3_5_lightning_30b_a3b`** (`nvidia/nemotron-3.5-lightning-30b-a3b`): Autonomous agent decision-making, structured JSON planning, and multi-step tool execution.
- **`glm_5_3_flash`** (`z-ai/glm-5.3-flash`): Structured reasoning, tool use, and agentic workflows.
- **`gemma_4_31b_it`** (`google/gemma-4-31b-it`): Deep coding, code analysis, debugging, and video analysis.

### 3. Intelligence Router
The built-in Intelligence Router deterministically selects the optimal model in $< 0.05$ms without requiring LLM pre-classifier calls:
- **Task Profiling**: Identifies task archetypes (`conversational`, `coding`, `agent_decision`, `job_step`, `video_analysis`) based on request metadata, prompt heuristics, and modality declarations.
- **7-Step Deterministic Selection Hierarchy**:
  1. Explicit Model Override Validation
  2. Required Capability Eligibility Filtering
  3. Required Modality Eligibility Filtering
  4. Canonical Task Specialization Matching
  5. Preferred Capability Overlap Scoring
  6. Structured-Output Agentic Precedence
  7. Static Deterministic Registry Order
- **Decoupled Fallback Candidates**: Supplies ordered fallback candidate IDs without silent network retries.

---

## Getting Started

### Prerequisites
- Python 3.10+
- An active NVIDIA API key with access to NVIDIA NIM / integrate endpoints

### Local Development Setup

1. **Clone the repository and enter the directory**:
   ```bash
   cd "Orange API"
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Environment Configuration

Copy the example environment file and configure your settings:
```bash
cp .env.example .env
```

Edit `.env` with your configuration:
```env
AI_PROVIDER=nvidia
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
DEFAULT_NVIDIA_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
```

> **Security Note**: Never commit `.env` or real API keys to version control. The `.gitignore` file is configured to exclude all environment and secret files by default.

### Running the API Locally

Start the local development server with Uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be accessible at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## API Endpoints

### Health Check
- **`GET /health`**
  Returns service status and active provider name.

### Main Inference Endpoint
- **`POST /assist`**
  Handles conversational and agentic assistance queries.

#### Request Body Schema
```json
{
  "message": "debug this Python function: def add(a, b): return a - b",
  "history": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello! How can I help you today?"}
  ],
  "profile": "Optional user context or instructions",
  "system_instruction": "Optional system prompt override",
  "provider": "nvidia",
  "model": "optional_model_key_override",
  "task_type": "optional_task_type",
  "modalities": ["text"]
}
```

#### Supported `task_type` values:
- `conversational`
- `coding`
- `agent_decision`
- `job_step`
- `video_analysis`

#### Supported `modalities`:
- `text` (default)
- `video`

#### Response Body Schema
```json
{
  "reply": "Here is the corrected Python function...",
  "model": "nvidia:google/gemma-4-31b-it",
  "success": true
}
```

---

## Running Tests

Run the unit and integration test suites:
```bash
python test_intelligence_router.py
python test_provider_architecture.py
```
