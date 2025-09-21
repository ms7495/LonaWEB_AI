# ui/streamlit_app_no_threading.py - Robust banner path handling (Windows-safe)

import hashlib
import os
import sys
import time
# Early warning suppression
import warnings
from pathlib import Path
from typing import Optional

import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup minimal logging for production
import logging

logging.basicConfig(level=logging.WARNING)

# Add paths for imports
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import your existing components
try:
    from core.rag_engine import DocuChatEngine
    from utils.file_utils import validate_file, get_file_hash
    import streamlit.components.v1 as components
except ImportError as e:
    st.error(f"[Import error] {e}")
    st.stop()

# --- Config ---
st.set_page_config(
    page_title="LonaWEB AI",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# --- Constants ---
ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.xlsx', '.xls', '.csv']
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
USER_UPLOAD_PREFIX = "user_upload_"


# ---------- Banner helpers ----------
def resolve_banner_path() -> Optional[Path]:
    """
    Try multiple likely locations for the banner image.
    Environment variable takes precedence: LONAWEB_BANNER
    """
    env_path = os.getenv("LONAWEB_BANNER")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))

    # Common repo locations
    candidates += [
        current_dir / "assets" / "lonaweb_banner.png",
        current_dir / "assets" / "banner.png",
        parent_dir / "assets" / "lonaweb_banner.png",
        parent_dir / "assets" / "banner.png",
        current_dir / "lonaweb_banner.png",
        current_dir / "banner.png",
    ]

    for p in candidates:
        try:
            if p and p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None


def render_header():
    """Render banner image if available; otherwise a clean text header."""
    banner_path = resolve_banner_path()
    if banner_path:
        # Use file bytes so Streamlit doesn't need to resolve container paths
        try:
            with open(banner_path, "rb") as f:
                st.markdown(
                    "<div style='display:flex;justify-content:center;padding:8px 0 4px 0;'></div>",
                    unsafe_allow_html=True,
                )
                st.image(f.read(), use_column_width=False)
                st.markdown(
                    "<p style='text-align:center;color:#888;margin-top:6px;'>Intelligent Document Analysis & Chat Platform</p>",
                    unsafe_allow_html=True,
                )
                return
        except Exception as e:
            st.warning(f"Could not load banner at '{banner_path}': {e}")

    # Fallback header (no image)
    st.markdown(
        """
        <div style='text-align:center;padding:18px 0 8px 0;'>
            <h1 style='margin:0;'>LonaWEB AI</h1>
            <p style='color:#888;margin:6px 0 0 0;'>Intelligent Document Analysis & Chat Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------

# --- Safe Engine Loading ---
@st.cache_resource(show_spinner=True)
def load_engine_safely():
    """Load the DocuChat engine"""
    try:
        engine = DocuChatEngine()
        return engine
    except Exception as e:
        st.error(f"[Init error] Failed to load LonaWEB AI engine: {e}")
        return None


# --- Utility Functions ---
def clean_document_name(source_name: str) -> str:
    """Clean up document names by removing prefixes"""
    if source_name.startswith(USER_UPLOAD_PREFIX):
        source_name = source_name[len(USER_UPLOAD_PREFIX):]

    import re
    uuid_pattern = r'^user_[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}_'
    source_name = re.sub(uuid_pattern, '', source_name)

    if source_name.startswith('user_'):
        source_name = source_name[5:]

    return source_name


# --- File Processing Functions ---
def process_uploaded_file(uploaded_file, progress_callback=None):
    """Process an uploaded file using the DocuChat engine"""
    try:
        if progress_callback:
            progress_callback(0.1, "Validating file...")

        if uploaded_file.size > MAX_FILE_SIZE:
            return {
                'success': False,
                'error': f'File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024:.0f}MB',
                'filename': uploaded_file.name
            }

        file_extension = Path(uploaded_file.name).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            return {
                'success': False,
                'error': f'Unsupported file type: {file_extension}. Supported: {", ".join(ALLOWED_EXTENSIONS)}',
                'filename': uploaded_file.name
            }

        existing_docs = st.session_state.engine.get_session_documents()
        for doc in existing_docs:
            if uploaded_file.name in doc.get('display_name', ''):
                return {
                    'success': False,
                    'error': f'A file named "{uploaded_file.name}" already exists. Delete it first or rename your file.',
                    'filename': uploaded_file.name
                }

        if progress_callback:
            progress_callback(0.3, "Processing document...")

        class MockUploadedFile:
            def __init__(self, file_obj):
                self.name = file_obj.name
                self.size = file_obj.size
                self._content = file_obj.getvalue()

            def getvalue(self):
                return self._content

            def read(self):
                return self._content

        mock_file = MockUploadedFile(uploaded_file)

        if progress_callback:
            progress_callback(0.5, "Generating embeddings...")

        result = st.session_state.engine.process_uploaded_file(mock_file)

        if progress_callback:
            progress_callback(0.9, "Finalizing...")

        if result.get("success", False):
            if progress_callback:
                progress_callback(1.0, "Processing complete.")

            return {
                'success': True,
                'filename': uploaded_file.name,
                'chunks': result.get('chunks_created', 0),
                'pages': result.get('pages', 0)
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'Unknown error occurred'),
                'filename': uploaded_file.name
            }

    except Exception as e:
        error_msg = str(e)
        if "No text content" in error_msg:
            error_msg = "Could not extract text from this file. It might be scanned or corrupted."
        elif "timeout" in error_msg.lower():
            error_msg = "Processing timeout. Try a smaller file or simpler document."
        elif "memory" in error_msg.lower():
            error_msg = "Out of memory. Try a smaller file."
        elif "'utf-8' codec can't decode" in error_msg:
            error_msg = "File encoding issue. Please ensure the file is not corrupted and is in a supported format."

        return {
            'success': False,
            'error': error_msg,
            'filename': uploaded_file.name
        }


def delete_user_document(doc_name):
    """Delete a user-uploaded document"""
    try:
        result = st.session_state.engine.clear_documents()
        return result.get('success', False)
    except Exception:
        return False


# --- Session State Management - SIMPLIFIED ---
def initialize_session_state():
    """Initialize session state with minimal complexity"""
    defaults = {
        "chat_history": [],
        "processing": False,
        "engine_loaded": False,
        "uploaded_file_key": 0,
        "file_processing": False,
        "current_mode": "LLM + Context",
        "form_key": 0,  # KEY FIX: Use form key to prevent state conflicts
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()

# --- Load Resources ---
with st.spinner("Initializing system..."):
    if not st.session_state.engine_loaded:
        engine = load_engine_safely()
        if engine:
            st.session_state.engine = engine
            st.session_state.engine_loaded = True
        else:
            if st.button("Retry Initialization"):
                load_engine_safely.clear()
                st.rerun()
            st.stop()
    else:
        engine = st.session_state.engine

# --- Sidebar ---
with st.sidebar:
    st.title("Settings & Files")

    st.subheader("Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'doc', 'txt', 'xlsx', 'xls', 'csv'],
        help=f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}. Max size: {MAX_FILE_SIZE / 1024 / 1024:.0f}MB",
        disabled=st.session_state.file_processing or st.session_state.processing,
        key=f"file_uploader_{st.session_state.uploaded_file_key}"
    )

    if uploaded_file is not None and not st.session_state.file_processing and not st.session_state.processing:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"File too large! Maximum size is {MAX_FILE_SIZE / 1024 / 1024:.0f}MB")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"{uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            with col2:
                if st.button("Process", key="process_file", type="primary"):
                    st.session_state.file_processing = True
                    st.rerun()

    # Handle file processing
    if st.session_state.file_processing and uploaded_file is not None:
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()


            def update_progress(progress, message):
                progress_bar.progress(progress)
                status_text.text(message)


            try:
                result = process_uploaded_file(uploaded_file, update_progress)
                if result['success']:
                    st.success(f"Successfully processed {result['filename']}")
                    st.caption(f"Created {result['chunks']} searchable chunks from {result['pages']} pages")
                    st.session_state.file_processing = False
                    st.session_state.uploaded_file_key += 1
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Failed to process file: {result['error']}")
                    st.session_state.file_processing = False
                    time.sleep(2)
                    st.rerun()
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                st.session_state.file_processing = False
                time.sleep(2)
                st.rerun()

            progress_bar.empty()
            status_text.empty()

    st.divider()
    st.subheader("Your Documents")
    try:
        user_docs = st.session_state.engine.get_session_documents()
        if user_docs:
            for doc in user_docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    display_name = doc.get('display_name', 'Unknown Document')
                    clean_name = clean_document_name(display_name)
                    display_name = clean_name[:30] + "..." if len(clean_name) > 30 else clean_name
                    chunks = doc.get('chunks', 0)
                    st.text(display_name)
                    st.caption(f"{chunks} chunks")
                with col2:
                    delete_key = f"del_{hashlib.md5(clean_name.encode()).hexdigest()[:8]}"
                    if st.button("Delete", key=delete_key,
                                 help="Delete document",
                                 disabled=st.session_state.file_processing or st.session_state.processing):
                        if delete_user_document(clean_name):
                            st.success("Deleted.")
                            time.sleep(1)
                            st.rerun()
        else:
            st.caption("No documents uploaded yet")
    except Exception as e:
        st.warning(f"Error loading documents: {e}")

    st.divider()
    st.subheader("Response Mode")
    mode = st.radio(
        "Choose how the AI responds:",
        ["LLM + Context", "LLM Only"],
        help="Context mode searches your documents for relevant information before responding. LLM Only uses the AI's general knowledge without document search.",
        index=0,
        disabled=st.session_state.processing
    )
    st.session_state.current_mode = mode
    st.caption("Will search your documents and provide contextual answers." if mode == "LLM + Context"
               else "Will respond using general AI knowledge only.")

    st.divider()
    if st.button("Clear Chat", use_container_width=True, disabled=st.session_state.processing):
        st.session_state.chat_history = []
        st.session_state.processing = False
        st.session_state.form_key += 1
        st.rerun()

# --- Main Interface (banner + subtitle, with fallback) ---
render_header()

# --- Chat Display ---
for i, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, #1e3a8a20, #1e40af20);
                    padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #3b82f6;'>
            <span style='color: #3b82f6; font-weight: bold; font-size: 0.9em;'>YOU</span><br>
            <span style='font-size: 1.05em;'>{msg['content']}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, #15803d20, #16a34a20);
                    padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #22c55e;'>
            <span style='color: #16a34a; font-weight: bold; font-size: 0.9em;'>LONAWEB AI</span><br>
            <div style='font-size: 1.05em; line-height: 1.6;'>{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)

# --- Show processing status if active ---
if st.session_state.processing:
    with st.spinner("Processing your question..."):
        time.sleep(0.1)

# --- Input Form - only when not processing ---
if not st.session_state.processing:
    with st.form(f"chat_form_{st.session_state.form_key}", clear_on_submit=True):
        st.markdown("**Ask about your documents or any topic:**")
        user_input = st.text_area(
            "Your question...",
            height=100,
            placeholder="What would you like to know about your documents?"
        )
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("Send Question", use_container_width=True)

        if submitted and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            st.session_state.processing = True
            st.rerun()

# --- Handle actual query processing AFTER the UI is set up ---
if st.session_state.processing and len(st.session_state.chat_history) > 0:
    last_msg = st.session_state.chat_history[-1]
    if last_msg["role"] == "user":
        user_question = last_msg["content"]
        try:
            status_placeholder = st.empty()
            with status_placeholder:
                st.info(
                    "Searching documents and generating response..." if st.session_state.current_mode == "LLM + Context"
                    else "Generating response...")

            recent_history = [m for m in st.session_state.chat_history[:-1] if m["role"] in ["user", "assistant"]][-6:]
            start_time = time.time()
            result = st.session_state.engine.query_documents(
                user_question=user_question,
                chat_history=recent_history,
                mode=st.session_state.current_mode
            )
            response_time = time.time() - start_time

            if result.get("success", False):
                reply = result.get("answer", "I couldn't generate a response.")
                sources = result.get("sources", [])
                context_used = result.get("context_used", 0)
                if sources and st.session_state.current_mode == "LLM + Context":
                    reply += "\n\n" + "=" * 50 + "\nSources Used:\n"
                    for source in sources:
                        reply += f"- {clean_document_name(source)}\n"

                mode_indicator = "Context Mode" if st.session_state.current_mode == "LLM + Context" else "LLM Only"
                reply += f"\n\nResponse Time: {response_time:.2f}s | Mode: {mode_indicator}"
                if st.session_state.current_mode == "LLM + Context":
                    reply += f" | Chunks Used: {context_used}"
            else:
                reply = ("I encountered an error while processing your question: " +
                         result.get('error', 'Unknown error') +
                         "\n\nTip: Rephrase your question or check if your documents contain the needed info.")

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.session_state.processing = False
            st.session_state.form_key += 1
            status_placeholder.empty()
            st.rerun()

        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": (
                    "I encountered an error while processing your question. "
                    "Please try again or rephrase your question.\n\n"
                    f"Error: {str(e)}\n\n"
                    "Tip: Make sure your documents are properly uploaded and try asking more specific questions."
                )
            })
            st.session_state.processing = False
            st.session_state.form_key += 1
            time.sleep(2)
            st.rerun()

# --- Auto-scroll ---
if st.session_state.chat_history:
    components.html("""
    <script>
        setTimeout(function() {
            window.parent.scrollTo(0, window.parent.document.body.scrollHeight);
        }, 300);
    </script>
    """, height=0)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>LonaWEB AI</strong> | Intelligent Document Analysis & Chat Platform</p>
    <p style='font-size: 0.8em; color: #888;'>Powered by advanced local AI models for secure document processing</p>
</div>
""", unsafe_allow_html=True)
