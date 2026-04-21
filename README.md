# Social-to-Lead Agentic Workflow (AutoStream AI)

📺 **[Click Here to Watch the Demo Video](https://drive.google.com/file/d/1XqG4zbr0Dz7bnr_C85MoNXME9eJaLYTz/view?usp=drivesdk)**

This is a real-world Conversational AI Agent built for **AutoStream**, a SaaS company providing automated video editing tools for content creators. The agent is constructed to intentionally identify user intent, answer product queries via a RAG (Retrieval-Augmented Generation) pipeline, and organically act on High-Intent leads via Tool Calls.

## 🚀 Core Capabilities
- **Intent Identification**: Effectively routes queries into *Casual greeting*, *Product/pricing inquiry*, or *High-intent lead*.
- **RAG-Powered Knowledge Retrieval**: A local Vectorstore containing company policies and pricing structures utilizing HuggingFace `sentence-transformers`.
- **Tool Execution (Lead Capture)**: Conditionally collects *Name*, *Email*, and *Creator Platform* before making a secure mock API call.
- **State Management**: Retains context across multiple conversational turns using localized LangGraph memory.

## 📂 Project Structure (6.1 Core Code)
The project strictly aligns with a modular design to separate logical concerns:
- `agent.py` → **Agent Logic & Intent Detection** (Orchestrates the LLM, sets up LangGraph state, loops interactions).
- `rag.py` → **RAG Pipeline** (Handles Markdown parsing, embeddings generation, and FAISS vectorstore integration).
- `tools.py` → **Tool Execution** (Defines external logic required for high-intent mock lead captures).
- `knowledge.md` → The local data source for retrieval indexing.

---

## 1. How to run the project locally

### Prerequisites
- Python 3.9+ installed
- A Google API Key (`GOOGLE_API_KEY`) from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Setup Database & Dependencies
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### Environment Variables
Create a locally-hosted `.env` file in the root directory to store your Google Gemini API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
*(Note: We utilize the ultra-fast `gemini-2.5-flash` model as the upgraded counterpart to the deprecated 1.5 version).*

### Run the Agent
Execute the main agent script to launch the interactive terminal session:
```bash
python agent.py
```

---

## 2. Architecture Explanation

For orchestrating the pipeline, I chose **LangGraph** paired with Google's Gemini models due to its seamless cyclical capabilities. Unlike standard linear chain execution, LangGraph provides fine-grained control over execution loops, enabling a strict **ReAct** (Reasoning and Acting) execution pattern where the agent can intelligently route to tools and recursively loop back to evaluate results.

**State & Memory Management:**
State is maintained intrinsically via LangGraph's checkpointer (`MemorySaver`). Every interactive message turn is appended to a unified messaging state tied to a distinct `thread_id` (e.g., `user_session_1`). LangGraph manages retaining the context window natively across the entirety of the dialogue, ensuring that the AI possesses the working memory of previously provided details (e.g., name, email) without forcing the user to arbitrarily repeat themselves prior to executing the `mock_lead_capture` tool.

---

## 3. WhatsApp Deployment Question

**Explain how you would integrate this agent with WhatsApp using Webhooks:**

To integrate this established agent with WhatsApp via Webhooks, I would orchestrate the following workflow using the **WhatsApp Business Cloud API**:
1. **Set up Webhook Server:** Deploy a webhook endpoint via a backend framework (like Flask or FastAPI) hosted on AWS or Render, and register this `URL` with Meta's developer dashboard to passively listen for `messages` events.
2. **Handle Inbound Listeners:** When a user dictates a message on WhatsApp, Meta securely posts a JSON payload to our registered endpoint. We systematically parse the user's phone number as their unique `thread_id` and extract their raw message content.
3. **Graph Processing:** The extracted string is fed directly into the LangGraph state (matching the phone number `thread_id` to retrieve preceding memory). The agent dynamically evaluates intent, fetches RAG data if needed, calls the Mock API if highly-interested, and issues a final string response.
4. **Send Outbound Delivery:** Instead of printing to a terminal, the backend takes the generated `Agent` response and fires a reverse HTTPS `POST` request back to the WhatsApp API `messages` endpoint to dispatch our text onto the user's physical device in real-time.
