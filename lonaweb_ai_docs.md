# LonaWEB AI - Project Documentation

## Project Overview

LonaWEB AI is an intelligent document analysis and chat platform that enables users to upload documents and interact with them through natural language queries. The system uses Retrieval-Augmented Generation (RAG) to provide contextual answers based on uploaded content.

## Features

- **Document Processing**: Support for PDF, DOCX, DOC, TXT, XLSX, XLS, and CSV files
- **Intelligent Chat**: AI-powered conversations with document context
- **Vector Search**: Semantic search through uploaded documents
- **Local AI Models**: Uses GGUF models for offline operation
- **Web Interface**: Streamlit-based user interface
- **API Support**: FastAPI backend for programmatic access

## Project Structure

```
PythonProject/
├── local_docs/                     # Default folder for local file processing
├── main/                          # Main application directory
│   ├── api/                       # FastAPI backend
│   ├── config/                    # Configuration files
│   ├── core/                      # Core business logic
│   ├── models/                    # AI model storage
│   ├── ui/                        # User interface components
│   ├── utils/                     # Utility functions
│   └── uploaded_docs/             # Runtime document storage
├── process_local_files.py         # Batch document processor
└── start_lonaweb_ai.py            # Application launcher
```

## Directory Details

### `/main/` - Main Application

The core application containing all modules and components.

### `/main/api/` - API Layer

**Files:**
- `__init__.py` - Package initialization
- `main.py` - FastAPI application with REST endpoints

**Purpose:** Provides REST API endpoints for document upload, querying, and management.

**Key Endpoints:**
- `POST /upload` - Upload and process documents
- `POST /query` - Query documents with natural language
- `GET /documents` - List uploaded documents
- `GET /model-info` - Get AI model information

### `/main/config/` - Configuration

**Files:**
- `__init__.py` - Package initialization
- `settings.py` - Application configuration and environment variables

**Purpose:** Centralized configuration management for the entire application.

**Key Settings:**
- File processing limits and supported formats
- AI model configuration
- Vector database settings
- API keys and URLs

### `/main/core/` - Core Business Logic

The heart of the application containing all main functionality.

#### `chat.py`
- **Purpose:** Chat session management and conversation history
- **Classes:** `ChatMessage`, `ChatHistory`, `ChatManager`
- **Features:** Message storage, context management, session handling

#### `document_processor.py`
- **Purpose:** Document text extraction and chunking
- **Supported Formats:** PDF, DOCX, TXT, Excel/CSV
- **Features:** Smart text extraction, content chunking, metadata generation

#### `embeddings.py`
- **Purpose:** Text embedding generation for semantic search
- **Features:** Local model support, Sentence Transformers integration
- **Models:** Supports various embedding models with auto-detection

#### `llm_provider.py`
- **Purpose:** Large Language Model integration with GPU support
- **Features:** GGUF model loading, GPU acceleration, chat completion
- **Models:** Llama, Mistral, and other GGUF-compatible models

#### `rag_engine.py`
- **Purpose:** Main RAG (Retrieval-Augmented Generation) orchestrator
- **Features:** Document processing pipeline, query handling, response generation
- **Classes:** `DocuChatEngine` (main engine class)

#### `search.py`
- **Purpose:** Document search and retrieval functionality
- **Features:** Semantic search, keyword search, hybrid search, result ranking

#### `vector_store.py`
- **Purpose:** Vector storage and similarity search
- **Features:** In-memory vector storage, cosine similarity, CRUD operations

### `/main/models/` - AI Model Storage

**Structure:**
```
models/
├── embeddings/                     # Embedding models
└── something.gguf                  # Main language model
```

**Purpose:** Storage for AI models used by the application.

**Model Types:**
- **GGUF Files:** Quantized language models for efficient inference
- **Embedding Models:** Sentence transformer models for text embeddings

### `/main/ui/` - User Interface

#### `streamlit_app.py`
- **Purpose:** Main Streamlit web application
- **Features:** File upload, chat interface, document management
- **Components:** Sidebar controls, chat display, progress tracking

#### `uploaded_docs/`
- **Purpose:** Runtime storage for uploaded documents
- **Note:** Temporary storage during processing

### `/main/utils/` - Utility Functions

#### `file_utils.py`
- **Purpose:** File validation and utility functions
- **Features:** File type validation, hash generation, filename cleaning

#### `text_utils.py`
- **Purpose:** Text processing utilities
- **Features:** Text cleaning, chunking, keyword extraction, highlighting

### Root Level Files

#### `process_local_files.py`
- **Purpose:** Batch processing tool for local documents
- **Usage:** Process entire folders of documents
- **Features:** Recursive processing, progress tracking, duplicate detection

#### `start_lonaweb_ai.py`
- **Purpose:** Main application launcher
- **Features:** Environment setup, model detection, Streamlit startup
- **Usage:** Primary entry point for the application

#### `run_app.py`
- **Purpose:** Alternative launcher with path setup
- **Features:** Python path configuration, Streamlit execution

## Getting Started

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install GPU-enabled llama-cpp-python (optional)
pip uninstall llama-cpp-python
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### 2. Model Setup

Place your GGUF language model in `/main/models/`:
```
main/models/Llama-3.2-1B-Instruct-f16.gguf
```

### 3. Start the Application

```bash
# Using the main launcher
python start_lonaweb_ai.py

# Or using the alternative launcher
python main/run_app.py
```

### 4. Process Local Documents (Optional)

```bash
# Process documents in local_docs folder
python process_local_files.py

# Process specific folder
python process_local_files.py /path/to/documents

# Show help
python process_local_files.py --examples
```

## Configuration

### Environment Variables

```bash
# LLM Configuration
LLAMA_MODEL_PATH=/path/to/model.gguf
LLAMA_N_GPU_LAYERS=32
LLAMA_USE_GPU=true
LLAMA_N_CTX=4096

# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION=documents

# Embedding Model
EMBEDDER_NAME=sentence-transformers/all-MiniLM-L6-v2

# Banner Image (optional)
LONAWEB_BANNER=/path/to/banner.png
```

### GPU Acceleration

For optimal performance with NVIDIA GPUs:

1. Install CUDA-enabled llama-cpp-python
2. Set `LLAMA_N_GPU_LAYERS=32` (or appropriate value)
3. Set `LLAMA_USE_GPU=true`

## Usage Workflows

### Document Upload and Chat
1. Start the application
2. Upload documents via the sidebar
3. Wait for processing completion
4. Ask questions about your documents
5. Switch between "LLM + Context" and "LLM Only" modes

### Batch Document Processing
1. Place documents in `local_docs/` folder
2. Run `python process_local_files.py`
3. Documents are processed and indexed
4. Start the main application to query

### API Usage
1. Start the FastAPI server: `python main/api/main.py`
2. Upload documents via POST `/upload`
3. Query documents via POST `/query`
4. Access API docs at `http://localhost:8000/docs`

## Dependencies

### Core Dependencies
- **streamlit** - Web interface framework
- **fastapi** - API framework
- **qdrant-client** - Vector database client
- **sentence-transformers** - Embedding models
- **llama-cpp-python** - GGUF model inference

### Document Processing
- **pdfplumber** - PDF text extraction
- **python-docx** - Word document processing
- **pandas** - Excel/CSV processing

### Optional Dependencies
- **torch** - GPU acceleration detection
- **uvicorn** - ASGI server for FastAPI

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure GGUF model is in `/main/models/`
2. **GPU not working**: Install CUDA-enabled llama-cpp-python
3. **Import errors**: Check Python path and dependencies
4. **Qdrant connection**: Ensure Qdrant server is running or use in-memory mode

### Logs and Debugging

The application uses Python logging. Check console output for detailed error messages and debugging information.

## Architecture

### Data Flow
1. **Document Upload** → Text Extraction → Chunking → Embedding → Vector Storage
2. **User Query** → Embedding → Vector Search → Context Retrieval → LLM Generation → Response

### Key Components
- **RAG Engine**: Orchestrates the entire pipeline
- **Vector Store**: Manages document embeddings and search
- **LLM Provider**: Handles AI model inference
- **Document Processor**: Extracts and chunks text from various formats

## License

This project is designed for document analysis and AI-powered chat functionality. Ensure you have appropriate licenses for any AI models you use.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review log output for error details
3. Verify model and dependency installation
4. Ensure proper configuration of environment variables