"""
ColBERT RAG Module - Token-level late interaction retrieval
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ColBERTDocument:
    """ColBERT用の文書データクラス"""
    content: str
    doc_id: int
    metadata: Dict
    embeddings: torch.Tensor = None  # トークンレベルの埋め込み


class ColBERTRAG:
    """
    ColBERT-based RAG with Token-level Late Interaction
    
    特徴:
    - トークンレベルマッチング（数値・固有名詞に強い）
    - MaxSim遅延相互作用
    - 2段階検索（mean pooling filter + MaxSim ranking）
    """
    
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        """
        Args:
            model_name: ColBERTモデル名
        """
        from transformers import AutoTokenizer, AutoModel
        
        logger.info(f"Initializing ColBERT RAG with {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # モデルロード
        logger.info("Loading ColBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # fp16最適化（GPU）
        if self.device == 'cuda':
            self.model.half()
            logger.info('Using fp16 for reduced memory')
        
        self.documents: List[ColBERTDocument] = []
        self.index_built = False
        
        logger.info(f"✓ ColBERT RAG initialized on {self.device}")
    
    def load_documents(
        self,
        data_dir: Path,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        sample_ratio: float = 0.5,
        max_length: int = 512
    ):
        """
        文書読み込みとチャンク分割
        
        Args:
            data_dir: データディレクトリ
            chunk_size: チャンクサイズ
            chunk_overlap: チャンクオーバーラップ
            sample_ratio: サンプリング比率
            max_length: 最大トークン長
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
        
        # ColBERTDocument作成
        self.documents = [
            ColBERTDocument(
                content=chunk.page_content[:max_length],
                doc_id=i,
                metadata=chunk.metadata
            )
            for i, chunk in enumerate(chunks)
        ]
        
        logger.info(f"✓ Loaded {len(self.documents)} documents")
    
    def encode_documents_batch(self, batch_size: int = 16):
        """
        文書をバッチでエンコード（トークンレベル埋め込み）
        
        Args:
            batch_size: バッチサイズ
        """
        logger.info(f"Encoding {len(self.documents)} documents...")
        
        for i in range(0, len(self.documents), batch_size):
            batch = self.documents[i:i + batch_size]
            texts = [doc.content for doc in batch]
            
            # トークナイズ
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            # 埋め込み取得
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
            
            # L2正規化
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=2)
            
            # CPU保存（メモリ節約）
            for j, doc in enumerate(batch):
                doc.embeddings = embeddings[j].cpu()
            
            # GPUキャッシュクリア
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            if (i // batch_size) % 10 == 0:
                logger.info(f"  Encoded {min(i + batch_size, len(self.documents))}/{len(self.documents)} documents")
        
        logger.info("✓ Document encoding complete")
    
    def build_index(self, batch_size: int = 16):
        """インデックス構築"""
        self.encode_documents_batch(batch_size=batch_size)
        self.index_built = True
        logger.info(f"✓ Index built with {len(self.documents)} documents")
    
    def compute_colbert_score(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> float:
        """
        ColBERT MaxSim スコア計算
        
        Args:
            query_embeddings: クエリ埋め込み (query_len, hidden_dim)
            doc_embeddings: 文書埋め込み (doc_len, hidden_dim)
        
        Returns:
            正規化されたMaxSimスコア
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # GPUに移動
        query_gpu = query_embeddings.to(device)
        doc_gpu = doc_embeddings.to(device)
        
        # 正規化
        query_norm = torch.nn.functional.normalize(query_gpu, p=2, dim=1)
        doc_norm = torch.nn.functional.normalize(doc_gpu, p=2, dim=1)
        
        # 類似度行列: (query_len, doc_len)
        similarity_matrix = torch.matmul(query_norm, doc_norm.T)
        
        # MaxSim: 各クエリトークンに対する最大類似度の合計
        max_similarities = similarity_matrix.max(dim=1)[0]
        
        # 正規化: クエリトークン数で割る（0-1スケール）
        num_query_tokens = query_embeddings.size(0)
        colbert_score = max_similarities.sum().item() / num_query_tokens
        
        return colbert_score
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        2段階検索: mean pooling filter + ColBERT MaxSim
        
        Args:
            query: 検索クエリ
            top_k: 取得件数
        
        Returns:
            [(content, score, metadata), ...]
        """
        if not self.index_built:
            raise ValueError("Index not built. Call build_index() first.")
        
        # クエリエンコード
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            query_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)
        
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        
        # Stage 1: Mean Pooling高速フィルタ（上位50候補）
        query_mean = query_embeddings.mean(dim=0).cpu()
        
        candidates = []
        for doc in self.documents:
            doc_mean = doc.embeddings.mean(dim=0)
            score = torch.dot(query_mean, doc_mean).item()
            candidates.append((doc, score))
        
        # 上位50候補
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:min(50, len(candidates))]
        
        # Stage 2: ColBERT MaxSim精密ランキング
        results = []
        for doc, _ in top_candidates:
            colbert_score = self.compute_colbert_score(query_embeddings, doc.embeddings)
            results.append((doc.content, colbert_score, doc.metadata))
        
        # スコア順ソート
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def get_stats(self) -> Dict:
        """統計情報取得"""
        return {
            "system_name": "ColBERT RAG",
            "model": self.model_name,
            "total_documents": len(self.documents),
            "index_built": self.index_built,
        }


def test_colbert_rag():
    """ColBERT RAGのテスト"""
    from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, COLBERT_SAMPLE_RATIO
    import time
    
    print("\n=== ColBERT RAG Test ===")
    
    # 初期化
    rag = ColBERTRAG()
    
    # 文書読み込み
    rag.load_documents(
        data_dir=DATA_DIR,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        sample_ratio=0.1  # テスト用10%
    )
    
    # インデックス構築
    rag.build_index(batch_size=16)
    
    # テストクエリ
    test_queries = [
        "堤防の天端幅は何メートルですか？",
        "コンクリートの設計基準強度は？",
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
    test_colbert_rag()
