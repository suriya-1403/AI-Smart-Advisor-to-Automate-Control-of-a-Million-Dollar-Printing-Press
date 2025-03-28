"""
Vector database management for the RAG Chatbot.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Optional

from config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL
from log import logger


class VectorDBManager:
    """
    Manages vector database operations including initialization and queries.

    This class encapsulates all interactions with the ChromaDB vector database,
    including initialization, document storage, and querying.
    """

    def __init__(
            self,
            chroma_path: str = CHROMA_PATH,
            collection_name: str = COLLECTION_NAME,
            embedding_model_name: str = EMBEDDING_MODEL
    ):
        """Initialize the vector database manager."""
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.initialize_db()

    def initialize_db(self) -> None:
        """Initialize or reset the ChromaDB instance."""
        # Ensure directory exists
        if not os.path.exists(self.chroma_path):
            os.makedirs(self.chroma_path, exist_ok=True)

        try:
            # Create a new ChromaDB instance
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name
            )
            logger.info("✅ ChromaDB store initialized successfully!")
            collection_count = self.collection.count()
            logger.info(f"Documents in collection: {collection_count}")
        except ValueError as e:
            logger.error(f"⚠️ Error initializing ChromaDB: {e}")
            # Force delete any corrupted files and retry
            os.system(f"rm -rf {self.chroma_path}")
            os.makedirs(self.chroma_path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            self.collection = self.chroma_client.get_or_create_collection(self.collection_name)
            logger.info("🔄 ChromaDB store reset and initialized!")

    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error counting collection: {e}")
            return 0

    def store_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Store multiple documents in the vector database."""
        for doc in documents:
            self.store_document(doc)

    def store_document(self, doc: Dict[str, Any]) -> None:
        """Store a single document in the vector database."""
        # Create a text representation of the document
        doc_text = f"Event {doc['event_id']} at {doc['location']} on {doc['publish_date']} with {doc['Ink Coverage']} ink and {doc['Media Weight GSM']} GSM."

        # Compute embeddings
        embedding = self.embedding_model.encode(doc_text).tolist()

        # Prepare metadata from document
        metadata = {
            "event_id": doc["event_id"],
            "publish_date": doc["publish_date"],
            "location": doc["location"],
            "Ink Coverage": doc["Ink Coverage"],
            "Media Weight GSM": doc["Media Weight GSM"],
            "Media Coating": doc["Media Coating"],
            "Media Finish": doc["Media Finish"],
            "Press Model": doc["Press Model"],
            "filename": doc.get("filename", "unknown")
        }

        # Add to ChromaDB
        self.collection.add(
            documents=[doc_text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[str(doc["event_id"])]
        )

    def query_by_embedding(
            self,
            query_embedding: List[float],
            where_clause: Optional[Dict[str, Any]] = None,
            n_results: int = 3
    ) -> Dict[str, Any]:
        """Query the vector database using embeddings and optional filters."""
        try:
            if where_clause and where_clause.get("$and", []):
                return self.collection.query(
                    query_embeddings=[query_embedding],
                    where=where_clause,
                    n_results=n_results
                )
            else:
                return self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
        except Exception as e:
            logger.error(f"Error in query: {e}")
            # Try a simpler query as fallback
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )