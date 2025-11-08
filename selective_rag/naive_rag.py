"""
Naive RAG Module - シンプルなFAISS + Sentence-Transformers実装
"""

import logging
from typing import List, Tuple, Dict
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class NaiveRAG:
    """
    ベースラインRAG実装
    FAISS + Sentence-Transformers (all-MiniLM-L6-v2)
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Sentence-Transformersモデル名
        """
        from sentence_transformers import SentenceTransformer
        import faiss
        
        logger.info(f"Initializing Naive RAG with {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.documents = []
        self.index = None
        self.model_name = model_name
        
    def load_documents(
        self,
        data_dir: Path,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        sample_ratio: float = 1.0
    ):
        """
        文書読み込みとチャンク分割
        
        Args:
            data_dir: データディレクトリ
            chunk_size: チャンクサイズ
            chunk_overlap: チャンクオーバーラップ
            sample_ratio: サンプリング比率（0.0-1.0）
        """
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        logger.info(f"Loading documents from {data_dir}")
        
        # Markdownファイル読み込み
        raw_docs = []
        for filepath in data_dir.glob("**/*.md"):
            try:
                loader = TextLoader(str(filepath), encoding='utf-8')
                raw_docs.extend(loader.load())
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        # チャンク分割
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(raw_docs)
        
        # サンプリング
        if sample_ratio < 1.0:
            import random
            sample_size = int(len(chunks) * sample_ratio)
            chunks = random.sample(chunks, sample_size)
            logger.info(f"📊 Sampled {len(chunks)} chunks ({sample_ratio*100:.0f}%)")
        
        self.documents = [(chunk.page_content, chunk.metadata) for chunk in chunks]
        logger.info(f"✓ Loaded {len(self.documents)} documents")
        
    def build_index(self):
        """FAISSインデックス構築"""
        import faiss
        
        logger.info("Building FAISS index...")
        
        texts = [doc[0] for doc in self.documents]
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        # Inner Product用のインデックス（正規化済み→コサイン類似度と等価）
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"✓ Built FAISS index with {self.index.ntotal} documents")
        
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        クエリ検索
        
        Args:
            query: 検索クエリ
            top_k: 取得件数
        
        Returns:
            [(content, score, metadata), ...] のリスト
        """
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
            if idx != -1:  # 有効なインデックス
                content, metadata = self.documents[idx]
                results.append((content, float(score), metadata))
        
        return results
    
    def get_stats(self) -> Dict:
        """統計情報取得"""
        return {
            "system_name": "Naive RAG",
            "model": self.model_name,
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
        }


def test_naive_rag():
    """Naive RAGのテスト"""
    from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    
    print("\n=== Naive RAG Test ===")
    
    # 初期化
    rag = NaiveRAG()
    
    # 文書読み込み
    rag.load_documents(
        data_dir=DATA_DIR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        sample_ratio=0.1  # テスト用に10%
    )
    
    # インデックス構築
    rag.build_index()
    
    # テストクエリ
    test_queries = [
        "堤防の天端幅は何メートルですか？",
        "河川管理とは何ですか？",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        start = time.time()
        results = rag.search(query, top_k=3)
        elapsed = (time.time() - start) * 1000
        
        print(f"Time: {elapsed:.2f}ms")
        for i, (content, score, metadata) in enumerate(results, 1):
            print(f"  {i}. Score: {score:.4f}")
            print(f"     {content[:100]}...")
    
    # 統計
    stats = rag.get_stats()
    print(f"\n{stats}")


if __name__ == "__main__":
    import time
    test_naive_rag()
