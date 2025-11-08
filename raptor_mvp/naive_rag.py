"""
Naive RAG Implementation
シンプルなベクトル検索RAG
"""
import numpy as np
import faiss
from typing import List, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import logging

from config import *

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class NaiveRAG:
    """シンプルなFAISSベクトル検索RAG"""
    
    def __init__(self):
        logger.info(f"Initializing Naive RAG with {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        self.chunks = []
        self.chunk_metadata = []
        self.index = None
        
    def load_documents(self, data_dir: Path) -> None:
        """文書を読み込みチャンク化"""
        logger.info(f"Loading documents from {data_dir}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )
        
        all_chunks = []
        all_metadata = []
        
        for file_path in data_dir.glob("*.md"):
            logger.info(f"  Processing {file_path.name}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = text_splitter.split_text(content)
            
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'source': file_path.name,
                    'chunk_id': idx,
                    'content_preview': chunk[:100]
                })
        
        self.chunks = all_chunks
        self.chunk_metadata = all_metadata
        logger.info(f"Loaded {len(self.chunks)} chunks from {len(list(data_dir.glob('*.md')))} documents")
    
    def build_index(self) -> None:
        """FAISSインデックスを構築"""
        logger.info("Building FAISS index...")
        start_time = time.time()
        
        # 埋め込み生成
        embeddings = self.embedding_model.encode(
            self.chunks,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # FAISSインデックス構築
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
        self.index.add(embeddings.astype('float32'))
        
        build_time = time.time() - start_time
        logger.info(f"Index built in {build_time:.2f}s ({len(self.chunks)} vectors, dim={dimension})")
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[str, float, dict]]:
        """クエリに対して検索"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # クエリ埋め込み
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # 検索
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((
                    self.chunks[idx],
                    float(score),
                    self.chunk_metadata[idx]
                ))
        
        return results
    
    def save(self, output_path: Path) -> None:
        """インデックスと

データを保存"""
        import pickle
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.chunk_metadata,
                'model_name': EMBEDDING_MODEL
            }, f)
        
        faiss.write_index(self.index, str(output_path.with_suffix('.faiss')))
        logger.info(f"Saved to {output_path}")
    
    def load(self, output_path: Path) -> None:
        """保存したインデックスを読み込み"""
        import pickle
        
        with open(output_path, 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data['chunks']
        self.chunk_metadata = data['metadata']
        self.index = faiss.read_index(str(output_path.with_suffix('.faiss')))
        logger.info(f"Loaded from {output_path}")


if __name__ == "__main__":
    # テスト実行
    rag = NaiveRAG()
    rag.load_documents(DATA_DIR)
    rag.build_index()
    
    # サンプルクエリ
    query = "堤防と護岸の機能的違いは？"
    results = rag.search(query, top_k=3)
    
    print(f"\nQuery: {query}\n")
    for i, (chunk, score, meta) in enumerate(results, 1):
        print(f"{i}. Score: {score:.4f} | Source: {meta['source']}")
        print(f"   {chunk[:150]}...\n")
    
    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rag.save(OUTPUT_DIR / "naive_rag.pkl")
