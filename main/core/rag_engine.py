# core/rag_engine.py - Production version with enhanced responses
import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import List, Dict

# Import updated modules
from core.document_processor import DocumentProcessor
from core.embeddings import EmbeddingSystem
from core.llm_provider import get_llm_provider
# External dependencies
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDER_NAME = os.getenv("EMBEDDER_NAME", "sentence-transformers/all-MiniLM-L6-v2")

UPLOAD_DIR = Path("uploaded_docs")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)


def find_models_directory():
    """Find the models directory from various possible locations"""
    possible_paths = [
        Path("models"),
        Path("main/models"),
        Path("../models"),
        Path("./main/models"),
    ]

    for path in possible_paths:
        if path.exists():
            logger.info(f"Found models directory at: {path.absolute()}")
            return path.absolute()

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Created models directory at: {models_dir.absolute()}")
    return models_dir.absolute()


MODELS_DIR = find_models_directory()


class DocuChatEngine:
    """Main RAG system with enhanced response generation"""

    def __init__(self):
        self.embedder = self._get_embedder()
        self.qdrant = self._get_qdrant()
        self.llm = self._get_llm()
        self.doc_processor = DocumentProcessor()
        self.session_docs = []
        self.processed_file_hashes = set()

        if not self.llm.is_available():
            logger.warning("GGUF model not loaded - check logs for details")

    def _get_embedder(self):
        """Load embedding model"""
        if not hasattr(self, '_embedder_cache'):
            logger.info(f"Loading embedding model: {EMBEDDER_NAME}")
            model = SentenceTransformer(EMBEDDER_NAME)
            _ = model.encode(["test"], normalize_embeddings=True)
            self._embedder_cache = model
            logger.info("✅ Embedding model loaded successfully")
        return self._embedder_cache

    def _get_qdrant(self):
        """Load Qdrant client with proper error handling"""
        if not hasattr(self, '_qdrant_cache'):
            logger.info(f"Connecting to Qdrant at: {QDRANT_URL}")
            try:
                # Import here to avoid scope issues
                from qdrant_client import QdrantClient

                # Try to connect to the actual Qdrant server
                client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

                # Test the connection by listing collections
                collections = client.get_collections()
                logger.info(f"✅ Qdrant client connected successfully to {QDRANT_URL}")
                logger.info(
                    f"Found {len(collections.collections)} collections: {[c.name for c in collections.collections]}")

                self._qdrant_cache = client

            except Exception as e:
                logger.error(f"❌ Qdrant connection failed: {e}")
                logger.error("This is a CRITICAL issue - your documents won't be accessible!")

                # For now, don't use fallback - we want to see the real issue
                raise Exception(f"Cannot connect to Qdrant at {QDRANT_URL}: {e}")

        return self._qdrant_cache

    def _get_llm(self):
        """Load LLM provider"""
        if not hasattr(self, '_llm_cache'):
            logger.info("Initializing LLM provider...")
            os.environ["MODELS_DIR"] = str(MODELS_DIR)
            self._llm_cache = get_llm_provider()

            if self._llm_cache.is_available():
                logger.info("✅ LLM provider loaded successfully")
                model_info = self._llm_cache.get_model_info()
                logger.info(f"Model: {model_info.get('model_path', 'Unknown')}")
            else:
                logger.warning("❌ LLM provider not available")
        return self._llm_cache

    def _get_file_hash(self, file_content: bytes) -> str:
        """Get hash of file content to detect duplicates"""
        return hashlib.md5(file_content).hexdigest()

    def process_uploaded_file(self, uploaded_file):
        """Process uploaded file with duplicate detection"""
        try:
            logger.info(f"Processing file: {uploaded_file.name}")

            file_content = uploaded_file.getvalue()
            file_hash = self._get_file_hash(file_content)

            if file_hash in self.processed_file_hashes:
                logger.warning(f"File {uploaded_file.name} already processed (duplicate detected)")
                return {
                    "success": False,
                    "error": f"File '{uploaded_file.name}' appears to be already processed (duplicate content detected)"
                }

            temp_path = UPLOAD_DIR / uploaded_file.name
            temp_path.write_bytes(file_content)

            chunks, metadata = self.doc_processor.process_file(
                str(temp_path),
                original_filename=uploaded_file.name
            )

            if not chunks:
                return {"success": False, "error": "Could not extract content from file"}

            logger.info(f"Created {len(chunks)} chunks from {uploaded_file.name}")

            texts = [c.get("text", "") for c in chunks]
            embeddings = self.embedder.encode(texts, normalize_embeddings=True)

            self._store_in_qdrant(chunks, embeddings, file_hash)

            doc_info = {
                "display_name": uploaded_file.name,
                "source": str(temp_path),
                "chunks": len(chunks),
                "pages": metadata.get("total_pages", 0),
                "processed_at": metadata.get("processed_at", ""),
                "file_hash": file_hash
            }
            self.session_docs.append(doc_info)
            self.processed_file_hashes.add(file_hash)

            temp_path.unlink(missing_ok=True)

            logger.info(f"✅ Successfully processed {uploaded_file.name}")
            return {
                "success": True,
                "chunks_created": len(chunks),
                "pages": metadata.get("total_pages", 0)
            }

        except Exception as e:
            logger.error(f"File processing failed: {e}")
            return {"success": False, "error": str(e)}

    def query_documents(self, user_question: str, chat_history: List[Dict] = None, mode: str = "LLM + Context"):
        """Query documents with proper mode handling"""
        logger.info(f"Processing query in {mode} mode: {user_question[:100]}...")

        try:
            if chat_history is None:
                chat_history = []

            recent_history = [msg for msg in chat_history if msg.get("role") in ["user", "assistant"]][-6:]

            if mode == "LLM + Context":
                return self._handle_context_mode(user_question, recent_history)
            else:
                return self._handle_llm_only_mode(user_question, recent_history)

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "success": False,
                "error": f"Error processing query: {str(e)}"
            }

    def _handle_context_mode(self, user_question: str, recent_history: List[Dict]) -> Dict:
        """Handle LLM + Context mode with enhanced responses"""
        try:
            logger.info("Starting context mode processing...")

            # Check if we have any documents
            try:
                collection_info = self.qdrant.get_collection(COLLECTION)
                logger.info(f"Collection '{COLLECTION}' found with {collection_info.points_count} documents")

                if collection_info.points_count == 0:
                    return {
                        "success": True,
                        "answer": "No documents have been uploaded yet. Please upload some documents first to start chatting with them.",
                        "sources": [],
                        "context_used": 0
                    }

            except Exception as e:
                logger.error(f"Could not check collection '{COLLECTION}': {e}")
                return {
                    "success": True,
                    "answer": "No documents have been uploaded yet. Please upload some documents first to start chatting with them.",
                    "sources": [],
                    "context_used": 0
                }

            # Generate query embedding
            logger.info("Generating query embedding...")
            try:
                query_vector = self.embedder.encode([user_question], normalize_embeddings=True)[0]
                logger.info(f"Query vector shape: {query_vector.shape}")
            except Exception as e:
                logger.error(f"Failed to generate query embedding: {e}")
                return {
                    "success": False,
                    "error": f"Failed to generate query embedding: {str(e)}"
                }

            # Search for relevant documents
            logger.info("Searching for relevant documents...")
            try:
                search_results = self.qdrant.search(
                    collection_name=COLLECTION,
                    query_vector=query_vector.tolist(),
                    limit=10,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=0.1
                )
                logger.info(f"Found {len(search_results)} search results with threshold 0.1")

                # If no results with 0.1, try with even lower threshold
                if not search_results:
                    logger.info("No results with 0.1 threshold, trying 0.0...")
                    search_results = self.qdrant.search(
                        collection_name=COLLECTION,
                        query_vector=query_vector.tolist(),
                        limit=10,
                        with_payload=True,
                        with_vectors=False,
                        score_threshold=0.0
                    )
                    logger.info(f"Found {len(search_results)} search results with threshold 0.0")

            except Exception as e:
                logger.error(f"Search failed: {e}")
                return {
                    "success": False,
                    "error": f"Document search failed: {str(e)}"
                }

            if not search_results:
                return {
                    "success": True,
                    "answer": "I couldn't find relevant information in your documents for this question. Try rephrasing your question or check if the information is in the uploaded documents.",
                    "sources": [],
                    "context_used": 0
                }

            # Build context from search results
            logger.info("Building context from search results...")
            context_parts = []
            sources = set()
            max_context_chars = 3500  # Increased for more detailed responses

            for i, result in enumerate(search_results):
                try:
                    text = result.payload.get("text", "")
                    filename = result.payload.get("filename", "unknown document")
                    score = getattr(result, 'score', 0.0)

                    logger.debug(f"Result {i + 1}: score={score:.3f}, filename={filename}, text_length={len(text)}")

                    if text and len(" ".join(context_parts + [text])) <= max_context_chars:
                        context_parts.append(text)
                        sources.add(filename)
                    else:
                        break
                except Exception as e:
                    logger.warning(f"Error processing search result {i}: {e}")
                    continue

            if not context_parts:
                return {
                    "success": True,
                    "answer": "I found some documents but couldn't extract useful content. Please try rephrasing your question.",
                    "sources": [],
                    "context_used": 0
                }

            context = "\n\n".join(context_parts)
            logger.info(f"Built context with {len(context_parts)} chunks from {len(sources)} sources")

            # Generate answer using LLM
            logger.info("Generating answer with LLM...")
            try:
                if self.llm.is_available():
                    answer = self._generate_answer_with_gguf(user_question, context, recent_history)
                else:
                    answer = self._create_enhanced_fallback_answer(user_question, context, list(sources))

                logger.info("Answer generated successfully")

                return {
                    "success": True,
                    "answer": answer,
                    "sources": list(sources),
                    "context_used": len(context_parts)
                }
            except Exception as e:
                logger.error(f"Answer generation failed: {e}")
                return {
                    "success": True,
                    "answer": self._create_enhanced_fallback_answer(user_question, context, list(sources)),
                    "sources": list(sources),
                    "context_used": len(context_parts)
                }

        except Exception as e:
            logger.error(f"Context mode failed: {e}")
            return {
                "success": False,
                "error": f"Context mode error: {str(e)}"
            }

    def _generate_answer_with_gguf(self, question: str, context: str, chat_history: List[Dict] = None) -> str:
        """Generate detailed answer with GGUF model"""
        try:
            logger.info("Creating messages for LLM...")

            # Enhanced system prompt for more detailed responses
            system_prompt = """You are LonaWEB AI, an expert document analysis assistant. Your task is to provide comprehensive, detailed, and well-structured answers based on the provided documents.

Guidelines for responses:
- Provide thorough, detailed explanations based on the document context
- Structure your response with clear sections and bullet points when appropriate
- Include specific details, examples, and explanations from the documents
- When possible, provide practical implications and applications
- Cite specific information and explain its significance
- If the context contains lists, procedures, or steps, present them clearly
- Aim for comprehensive responses that fully address the user's question
- Use the document information to provide expert-level insights

Response structure:
1. Direct comprehensive answer
2. Detailed explanation with supporting information
3. Specific examples or applications from the documents
4. Additional relevant context that might be helpful
5. Clear organization with headers or bullet points when appropriate

Make your responses informative, professional, and comprehensive while staying strictly within the provided context."""

            messages = [{"role": "system", "content": system_prompt}]

            # Add recent chat history
            if chat_history:
                recent_history = chat_history[-4:]
                for msg in recent_history:
                    if msg.get("role") in ["user", "assistant"] and len(msg.get("content", "")) > 0:
                        messages.append({
                            "role": msg["role"],
                            "content": msg["content"][:500]
                        })

            # Create the user message with context
            user_content = f"""Based on the following document excerpts, please provide a comprehensive and detailed answer to the question. Use all relevant information from the context to give a thorough response.

DOCUMENT CONTEXT:
{context[:3000]}

QUESTION: {question}

Please provide a detailed, well-structured answer that:
1. Directly addresses the question comprehensively
2. Uses specific information from the documents
3. Explains relevant details and implications
4. Organizes the information clearly
5. Provides practical context where applicable

Answer based only on the information provided above, but make it as detailed and informative as possible."""

            messages.append({"role": "user", "content": user_content})

            logger.info(f"Prepared {len(messages)} messages for LLM")

            # Generate response with enhanced parameters
            try:
                if hasattr(self.llm, 'create_chat_completion'):
                    result = self.llm.create_chat_completion(
                        messages=messages,
                        max_tokens=1200,
                        temperature=0.3,
                        top_p=0.9,
                        stop=["</s>", "<|end|>", "<|eot_id|>", "Human:", "Context:", "QUESTION:"]
                    )
                    answer = result["choices"][0]["message"]["content"].strip()

                elif hasattr(self.llm, 'generate_chat'):
                    answer = self.llm.generate_chat(
                        messages=messages,
                        max_tokens=1200,
                        temperature=0.3,
                        top_p=0.9,
                        stop=["</s>", "<|end|>", "<|eot_id|>", "Human:", "Context:", "QUESTION:"]
                    )

                else:
                    prompt = f"""Based on this context from documents, provide a comprehensive, detailed answer:

Context: {context[:2000]}

Question: {question}

Provide a thorough, well-structured answer with specific details from the context:"""

                    answer = self.llm.generate(
                        prompt=prompt,
                        max_tokens=1200,
                        temperature=0.3,
                        stop=["</s>", "<|end|>", "Question:", "Context:"]
                    )

                # Clean and validate the answer
                if isinstance(answer, str):
                    answer = answer.strip()
                else:
                    answer = str(answer).strip()

                # Remove any artifacts
                for artifact in ["</s>", "<|end|>", "<|eot_id|>", "Human:", "Assistant:", "QUESTION:", "Context:"]:
                    answer = answer.replace(artifact, "").strip()

                if len(answer) < 50:
                    logger.warning("Generated answer too short, using enhanced fallback")
                    return self._create_enhanced_fallback_answer(question, context, [])

                logger.info(f"Successfully generated detailed answer of length: {len(answer)}")
                return answer

            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                return self._create_enhanced_fallback_answer(question, context, [])

        except Exception as e:
            logger.error(f"Answer generation setup failed: {e}")
            return self._create_enhanced_fallback_answer(question, context, [])

    def _handle_llm_only_mode(self, user_question: str, recent_history: List[Dict]) -> Dict:
        """Handle LLM Only mode with enhanced responses"""
        try:
            model_info = self.llm.get_model_info()
            if not model_info.get('is_loaded', False):
                return {
                    "success": False,
                    "error": "AI model is not available for LLM-only responses. Please check if your GGUF model is properly loaded."
                }

            # Enhanced system prompt for more detailed LLM-only responses
            messages = [
                {
                    "role": "system",
                    "content": """You are LonaWEB AI, a knowledgeable and helpful AI assistant. Provide comprehensive, detailed, and well-structured responses based on your knowledge.

Guidelines for responses:
- Give thorough, informative answers that fully address the question
- Structure your response clearly with sections, bullet points, or numbered lists when appropriate
- Provide context, examples, and practical applications when relevant
- Explain concepts in detail while remaining accessible
- Include relevant background information that helps understand the topic
- When discussing complex topics, break them down into understandable components
- Aim for responses that are both comprehensive and practical
- Use professional language while being conversational and engaging

Make your responses detailed, informative, and valuable to the user."""
                }
            ]

            for msg in recent_history:
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            messages.append({"role": "user", "content": user_question})

            try:
                logger.info("Generating detailed LLM-only response...")

                if hasattr(self.llm, 'create_chat_completion'):
                    llm_result = self.llm.create_chat_completion(
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1200,
                        top_p=0.9
                    )
                    answer = llm_result["choices"][0]["message"]["content"].strip()
                elif hasattr(self.llm, 'generate_chat'):
                    answer = self.llm.generate_chat(
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1200,
                        top_p=0.9
                    )
                else:
                    prompt = f"""You are LonaWEB AI, a helpful and knowledgeable assistant. Please provide a comprehensive, detailed answer to this question:

Question: {user_question}

Provide a thorough response with detailed explanations, examples, and practical information where relevant."""

                    answer = self.llm.generate(
                        prompt=prompt,
                        temperature=0.7,
                        max_tokens=1200,
                        top_p=0.9
                    )

                if not answer or len(answer.strip()) < 20:
                    raise ValueError("Generated answer is too short or empty")

                return {
                    "success": True,
                    "answer": answer.strip(),
                    "sources": [],
                    "context_used": 0
                }

            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                return {
                    "success": False,
                    "error": f"LLM generation failed: {str(e)}"
                }

        except Exception as e:
            logger.error(f"LLM-only mode error: {e}")
            return {
                "success": False,
                "error": f"LLM-only mode error: {str(e)}"
            }

    def _create_enhanced_fallback_answer(self, question: str, context: str, sources: List[str]) -> str:
        """Create enhanced fallback answer when LLM is not available"""

        # Create a more structured presentation of the context
        display_context = context[:2500] + "..." if len(context) > 2500 else context

        # Try to structure the context better
        paragraphs = [p.strip() for p in display_context.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            structured_context = '\n\n'.join(f"**Section {i + 1}:**\n{p}" for i, p in enumerate(paragraphs[:4]))
        else:
            structured_context = display_context

        answer = f"""**Based on your document analysis:**

{structured_context}

---

**Question Asked:** {question}

**Summary:** The above information from your documents contains relevant details that address your question. The content includes specific information about the topic you're inquiring about.

**Note:** This is a direct presentation of relevant document content. For AI-generated analysis and answers, please ensure your GGUF model is properly configured.

**Document Sources:** {', '.join(sources) if sources else 'Document excerpts shown above'}"""

        return answer

    def _store_in_qdrant(self, chunks, embeddings, file_hash):
        """Store chunks in Qdrant vector database"""
        try:
            # Check if collection exists
            try:
                collection_info = self.qdrant.get_collection(COLLECTION)
                logger.info(f"Using existing collection: {COLLECTION}")
            except:
                vector_size = len(embeddings[0])
                logger.info(f"Creating new collection: {COLLECTION} (dimension: {vector_size})")
                self.qdrant.create_collection(
                    collection_name=COLLECTION,
                    vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
                )

            # Create points for insertion
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                point_id = str(uuid.uuid4())
                payload = {
                    "text": chunk.get("text", ""),
                    "filename": chunk.get("original_filename", "unknown"),
                    "page_number": chunk.get("page_number", 1),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "file_hash": file_hash
                }

                points.append(qm.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                ))

            # Batch insert
            batch_size = 50
            total_batches = (len(points) + batch_size - 1) // batch_size

            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                batch_num = (i // batch_size) + 1

                logger.info(f"Inserting batch {batch_num}/{total_batches} ({len(batch)} points)")

                self.qdrant.upsert(
                    collection_name=COLLECTION,
                    points=batch,
                    wait=True
                )

            logger.info(f"✅ Successfully stored {len(points)} chunks in Qdrant")

        except Exception as e:
            logger.error(f"Qdrant storage failed: {e}")
            raise

    def get_session_documents(self):
        """Get list of documents in current session"""
        return self.session_docs.copy()

    def get_model_info(self):
        """Get information about the loaded model"""
        info = self.llm.get_model_info()

        try:
            collection_info = self.qdrant.get_collection(COLLECTION)
            info["documents_count"] = collection_info.points_count
        except:
            info["documents_count"] = 0

        info["embedding_model"] = EMBEDDER_NAME
        info["session_documents"] = len(self.session_docs)
        info["models_directory"] = str(MODELS_DIR)

        return info

    def clear_documents(self):
        """Clear all documents from the current session"""
        try:
            self.qdrant.delete_collection(COLLECTION)
            logger.info("✅ Cleared Qdrant collection")

            self.session_docs.clear()
            self.processed_file_hashes.clear()
            logger.info("✅ Cleared session documents")

            return {"success": True, "message": "All documents cleared"}

        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
            return {"success": False, "error": str(e)}

    def get_document_stats(self):
        """Get statistics about processed documents"""
        try:
            collection_info = self.qdrant.get_collection(COLLECTION)
            total_chunks = collection_info.points_count
        except:
            total_chunks = 0

        return {
            "total_documents": len(self.session_docs),
            "total_chunks": total_chunks,
            "documents": [
                {
                    "name": doc["display_name"],
                    "chunks": doc.get("chunks", 0),
                    "pages": doc.get("pages", 0),
                    "processed_at": doc.get("processed_at", ""),
                    "file_hash": doc.get("file_hash", "")
                }
                for doc in self.session_docs
            ]
        }
