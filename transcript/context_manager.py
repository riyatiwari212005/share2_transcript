import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextManager:
    def __init__(self, db_path: str = None, transcript_dir: str = None):
        self.db_path = db_path or settings.VECTOR_DB_PATH
        self.transcript_dir = transcript_dir or settings.TRANSCRIPT_DIR
        
        os.makedirs(self.db_path, exist_ok=True)
        
        logger.info("Initializing ChromaDB and embedding model")
        self.client = chromadb.PersistentClient(path=self.db_path)
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        try:
            self.collection = self.client.get_collection("video_transcripts")
            logger.info("Loaded existing collection")
        except:
            self.collection = self.client.create_collection(
                name="video_transcripts",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Created new collection")
    
    def embed_text(self, text: str) -> List[float]:
        return self.embedding_model.encode(text).tolist()
    
    def index_transcripts(self):
        transcript_files = list(Path(self.transcript_dir).glob("*.json"))
        transcript_files = [f for f in transcript_files if f.stem != "transcription_summary"]
        
        logger.info(f"Indexing {len(transcript_files)} transcripts")
        
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        doc_id = 0
        
        for file_path in transcript_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                video_name = data["video_name"]
                
                for segment in data["segments"]:
                    text = segment["text"].strip()
                    if len(text) > 20:
                        documents.append(text)
                        embeddings.append(self.embed_text(text))
                        metadatas.append({
                            "video_name": video_name,
                            "start_time": segment["start"],
                            "end_time": segment["end"]
                        })
                        ids.append(f"doc_{doc_id}")
                        doc_id += 1
                
            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
        
        if documents:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Indexed {len(documents)} segments")
        else:
            logger.warning("No documents to index")
    
    def search_context(self, query: str, n_results: int = 5) -> Dict:
        query_embedding = self.embed_text(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        if not results["documents"] or not results["documents"][0]:
            return {
                "found": False,
                "context": "",
                "sources": []
            }
        
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        relevant_docs = []
        sources = []
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            similarity = 1 - dist
            if similarity >= settings.SIMILARITY_THRESHOLD:
                relevant_docs.append(doc)
                sources.append({
                    "video": meta["video_name"],
                    "timestamp": f"{meta['start_time']:.2f}s - {meta['end_time']:.2f}s",
                    "similarity": similarity
                })
        
        if not relevant_docs:
            return {
                "found": False,
                "context": "",
                "sources": []
            }
        
        context = "\n\n".join(relevant_docs)
        
        if len(context) > settings.MAX_CONTEXT_LENGTH:
            context = context[:settings.MAX_CONTEXT_LENGTH] + "..."
        
        return {
            "found": True,
            "context": context,
            "sources": sources
        }
    
    def is_query_relevant(self, query: str) -> bool:
        result = self.search_context(query, n_results=3)
        return result["found"]

def main():
    context_manager = ContextManager()
    context_manager.index_transcripts()
    
    logger.info("Context indexing complete")
    
    test_query = "What is discussed in the videos?"
    result = context_manager.search_context(test_query)
    logger.info(f"Test query result: {result['found']}")

if __name__ == "__main__":
    main()
