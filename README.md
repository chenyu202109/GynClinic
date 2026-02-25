# GynClinic 🏥

**Development and Validation of a Traceable Reasoning Multi-Agent Framework for Simulating Real-World Gynecological Clinical Diagnosis**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

---

## Overview

GynClinic is a multi-agent framework that simulates real-world gynecological outpatient clinical diagnosis workflows. The system orchestrates multiple LLM-powered agents — including a **Doctor Interviewer**, a **Patient Simulator**, a **Summary Expert**, and an **Auxiliary Examination Agent** — to collaboratively complete the entire clinical reasoning process from initial patient interview to final revised diagnosis.

The framework features a **Retrieval-Augmented Generation (RAG)** pipeline with traceable citations, ensuring that every diagnostic recommendation is grounded in evidence from medical guidelines and literature.

## Key Features

- 🩺 **Multi-Agent Collaboration** — Four specialized agents simulate a realistic outpatient encounter through role-based interactions.
- 🔍 **RAG with Traceable Citations** — Integrates ChromaDB vector retrieval, Cohere reranking, and DSPy-powered citation faithfulness checking for transparent, evidence-based reasoning.
- 🛠️ **Tool-Augmented Diagnosis** — Agents can autonomously invoke external tools including Google Search, PubMed literature query, and patient examination report retrieval.
- 📋 **Two-Stage Diagnostic Pipeline** — Preliminary differential diagnosis followed by a revised diagnosis after auxiliary examination results, mirroring real clinical workflows.
- 🤖 **Doctor–Patient Dialogue Simulation** — Realistic multi-turn conversations where the patient agent gradually reveals symptoms, just like a real outpatient visit.

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                         GynClinic Framework                       │
│                                                                   │
│  ┌─────────────┐    Multi-turn     ┌──────────────────┐           │
│  │   Patient    │◄────Dialogue────►│ Doctor Interviewer│           │
│  │  Simulator   │                  │    (GynAgent)     │           │
│  └─────────────┘                   └────────┬─────────┘           │
│                                             │                     │
│                               ┌─────────────▼──────────────┐     │
│                               │     Summary Expert         │     │
│                               │  (Medical Record Summary)  │     │
│                               └─────────────┬──────────────┘     │
│                                             │                     │
│  ┌──────────────────────────────────────────▼────────────────┐   │
│  │              Stage 1: Preliminary Diagnosis               │   │
│  │  GynAgent + RAG + Tools (Google, PubMed)                  │   │
│  │  → 5 Differential Diagnoses + 8 Examination Items         │   │
│  └──────────────────────────────────┬────────────────────────┘   │
│                                     │                             │
│  ┌──────────────────────────────────▼────────────────────────┐   │
│  │              Auxiliary Examination Agent                   │   │
│  │  Retrieves lab results, imaging reports, etc.             │   │
│  └──────────────────────────────────┬────────────────────────┘   │
│                                     │                             │
│  ┌──────────────────────────────────▼────────────────────────┐   │
│  │              Stage 2: Revised Diagnosis                   │   │
│  │  GynAgent + RAG + Tools + Examination Results             │   │
│  │  → Final 5 Diagnoses with Confidence Scores               │   │
│  └───────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
```

## Diagnostic Workflow

The system executes a complete clinical diagnosis pipeline for each patient case:

### Phase 1: Doctor–Patient Interview
The **Doctor Interviewer** agent conducts a multi-turn conversation with the **Patient Simulator** agent, gradually collecting symptoms, medical history, menstrual/marital history, and family history — mimicking a real outpatient consultation.

### Phase 2: Medical Record Summarization
The **Summary Expert** processes the dialogue history and generates a structured medical record including chief complaint, present medical history, past history, menstrual/marital history, and family history.

### Phase 3: Preliminary Diagnosis (Stage 1)
The **GynAgent** (Doctor Diagnosis Agent) analyzes the summarized medical record using:
- **RAG pipeline** — Retrieves and reranks relevant medical guidelines from ChromaDB
- **External tools** — Searches Google and PubMed for up-to-date clinical evidence
- Outputs **5 differential diagnoses** with confidence scores and **8 recommended examination items**

### Phase 4: Auxiliary Examinations
The **Auxiliary Examination Agent** retrieves the patient's actual examination results (lab tests, imaging, etc.) based on the examination items recommended in Stage 1.

### Phase 5: Revised Diagnosis (Stage 2)
The **GynAgent** integrates all available information — medical history, physical examination, and auxiliary examination results — to produce a **revised diagnosis** with updated confidence scores and detailed clinical reasoning.

## Tech Stack

| Component              | Technology                                        |
|------------------------|---------------------------------------------------|
| **LLM**                | OpenAI GPT-4o                                     |
| **Agent Framework**    | LlamaIndex (OpenAI Agent / FunctionCallingAgent)  |
| **RAG Framework**      | DSPy                                              |
| **Vector Database**    | ChromaDB                                          |
| **Embeddings**         | OpenAI `text-embedding-3-large`                   |
| **Reranking**          | Cohere `rerank-english-v3.0`                      |
| **Citation Checking**  | DSPy ChainOfThought (faithfulness verification)   |
| **External Tools**     | Google Custom Search API, PubMed / arXiv           |

## Project Structure

```
GynClinic/
├── .env                          # API keys (OpenAI, Cohere, Google)
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies
├── README.md
└── src/
    ├── 0run.py                   # Main entry: single-threaded pipeline
    ├── 0LLM_duibi_thread.py      # Main entry: multi-threaded pipeline
    ├── med_agent.py              # MedOpenAIAgent (Doctor Diagnosis Agent)
    ├── rag.py                    # RAG pipeline (retrieval, rerank, generation)
    ├── rag_config.py             # RAG configuration (models, chunking, etc.)
    ├── signatures.py             # DSPy signatures (Search, Answer, Citations)
    ├── agent_tools.py            # Tool definitions (Google, PubMed, Exams)
    ├── get_mytools.py            # Tool selector utility
    ├── chroma_db_retriever.py    # ChromaDB retriever (DSPy-compatible)
    ├── citations_utils.py        # Citation creation & faithfulness checking
    ├── embed.py                  # Embedding script for knowledge base
    ├── preprocess_sources.py     # Data preprocessing (add source IDs)
    ├── rag_utils.py              # RAG utility functions
    ├── utils.py                  # General utilities
    ├── loguru_logger.py          # Logging configuration
    ├── rag_logger.py             # RAG-specific logging
    ├── preprocess_logger.py      # Preprocessing logging
    ├── Imaging/
    │   └── data.json             # Patient case data (input)
    ├── complete_oncology_data/
    │   └── meditron.jsonl        # Medical knowledge base (JSONL)
    ├── process_data/
    │   ├── deduplicate_data.py   # Data deduplication
    │   ├── filter_data_sources.py # Data source filtering
    │   └── scrape_meditron.py    # Knowledge base scraping
    └── results/                  # Output directory
        ├── dp_history/           # Doctor-patient dialogue logs
        ├── rag_results/          # RAG retrieval results
        ├── tool_results/         # Tool execution results
        ├── examinations/         # Examination results
        └── reference/            # Reference documents
```

## Installation

### Prerequisites

- Python 3.11+
- [ChromaDB](https://docs.trychroma.com/) server running locally
- API keys for OpenAI, Cohere, and Google Custom Search

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/chenyu202109/GynClinic.git
   cd GynClinic
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Copy the `.env` file and fill in your API keys:

   ```env
   OPENAI_BASE_URL="your-openai-base-url"
   OPENAI_API_BASE="your-openai-api-base"
   OPENAI_API_KEY="your-openai-api-key"
   COHERE_API_KEY="your-cohere-api-key"
   GOOGLE_API_KEY="your-google-api-key"
   GOOGLE_SEARCH_ENGINE="your-google-search-engine-id"
   ```

4. **Start ChromaDB server**

   ```bash
   chroma run --path ./src/chroma_db_oncology
   ```

5. **Build the knowledge base** (first time only)

   ```bash
   cd src
   # Preprocess data sources
   python preprocess_sources.py -d complete_oncology_data

   # Create embeddings and index
   python embed.py --to_embed meditron
   ```

## Usage

### Run the Diagnostic Pipeline

```bash
cd src
python 0run.py
```

This will:
1. Load patient cases from `Imaging/data.json`
2. For each patient, execute the full diagnostic workflow (interview → summary → preliminary diagnosis → auxiliary exams → revised diagnosis)
3. Save results to `Imaging/result.json`

### Multi-threaded Execution

For batch processing with multi-threading support:

```bash
cd src
python 0LLM_duibi_thread.py
```

### Patient Data Format

Patient cases are stored in JSON format (`Imaging/data.json`):

```json
{
    "patient_id": "W-2",
    "basic_msg": "**Basic information**: Female, 28 years old, system ID is W-2",
    "context_msg": "**Basic information**: ... **Chief complaint**: ... **Medical history**: ...",
    "physical_examination": "**Physical examination**: ...",
    "check_report": {
        "Vaginal pH": "> 6",
        "Trichomonas": "Detected",
        ...
    },
    "groud_truth": ["Trichomonas Vaginitis"]
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This project builds upon and adapts code from the following open-source projects:

- [LlamaIndex](https://github.com/run-llama/llama_index) — Agent framework (MIT License)
- [DSPy](https://github.com/stanfordnlp/dspy) — RAG pipeline and citation checking (MIT License)
- [ChromaDB](https://github.com/chroma-core/chroma) — Vector database