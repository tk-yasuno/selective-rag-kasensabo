"""
ColBERT RAG Implementation for Kasensabo
河川砂防ダム技術基準用ColBERT検索システム

特徴:
- トークンレベルのマッチング（数値・用語に強い）
- 遅延相互作用（Late Interaction）
- 橋梁診断で+10-15%の実績
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from dataclasses import dataclass
import pickle
import json

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel

from config import *

logging.basicConfig(level=LOG_LEVEL, format='%(levelname)s:%(name)s:%(message)s')
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
    ColBERT-based Retrieval Augmented Generation
    
    ColBERTの遅延相互作用を活用した検索システム。
    単一ベクトルではなく、トークン列全体で類似度を計算。
    """
    
    def __init__(self, model_name: str = COLBERT_MODEL):
        """
        初期化
        
        Args:
            model_name: ColBERTモデル名（HuggingFace）
        """
        logger.info(f"Initializing ColBERT RAG with {model_name}")
        
        self.device = DEVICE
        self.model_name = model_name
        
        # ColBERTモデルとトークナイザーのロード
        logger.info("Loading ColBERT model and tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            # Use fp16 on GPU to reduce memory if available
            if self.device == 'cuda':
                try:
                    self.model.half()
                    logger.info('Using model.half() for reduced memory (fp16)')
                except Exception:
                    logger.debug('model.half() not supported for this model')
            logger.info("✓ ColBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ColBERT model: {e}")
            logger.info("Falling back to BERT-base model for demonstration")
            # フォールバック: 標準BERTモデル
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            self.model = AutoModel.from_pretrained("bert-base-multilingual-cased").to(self.device)
            self.model.eval()
        
        self.documents: List[ColBERTDocument] = []
        self.index_built = False
        
    def load_documents(self, data_dir: Path = DATA_DIR) -> int:
        """
        文書を読み込み、チャンクに分割
        
        Args:
            data_dir: データディレクトリ
            
        Returns:
            読み込んだチャンク数
        """
        logger.info(f"Loading documents from {data_dir}")
        
        # Markdownファイルをシンプルに読み込み
        from langchain_community.document_loaders import TextLoader
        
        raw_docs = []
        for filepath in Path(data_dir).glob("**/*.md"):
            try:
                loader = TextLoader(str(filepath), encoding='utf-8')
                raw_docs.extend(loader.load())
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        logger.info(f"Loaded {len(raw_docs)} markdown files")
        
        # テキスト分割
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )
        
        chunks = splitter.split_documents(raw_docs)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # サンプリング適用
        if SAMPLE_RATIO < 1.0:
            import random
            sample_size = int(len(chunks) * SAMPLE_RATIO)
            chunks = random.sample(chunks, sample_size)
            logger.info(f"📊 Sampled {len(chunks)} chunks ({SAMPLE_RATIO*100:.0f}% of total)")
        
        # ColBERTDocument形式に変換
        for idx, chunk in enumerate(chunks):
            doc = ColBERTDocument(
                content=chunk.page_content[:MAX_DOCUMENT_LENGTH * 4],  # 安全マージン
                doc_id=idx,
                metadata=chunk.metadata
            )
            self.documents.append(doc)
        
        logger.info(f"Loaded {len(self.documents)} documents")
        return len(self.documents)
    
    def encode_document(self, text: str) -> torch.Tensor:
        """
        文書をトークンレベルで埋め込み
        
        Args:
            text: 文書テキスト
            
        Returns:
            トークンレベルの埋め込みテンソル (seq_len, hidden_dim)
        """
        # トークナイズ
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_DOCUMENT_LENGTH,
            padding="max_length"
        ).to(self.device)
        
        # 埋め込み生成
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 最終層の隠れ状態を使用
            embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
        
        return embeddings

    def encode_documents_batch(self, texts: List[str], batch_size: int = 8) -> List[torch.Tensor]:
        """
        複数文書をバッチ処理でエンコードしてトークン埋め込みを返す。
        すべての出力はCPUに移し、メモリ消費を分散する。
        """
        results: List[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=MAX_DOCUMENT_LENGTH,
                    padding='max_length',
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
                # move to cpu and append per-document tensor
                emb_cpu = emb.cpu()
                for b in range(emb_cpu.size(0)):
                    results.append(emb_cpu[b])
                # clear GPU cache between batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return results
    
    def encode_query(self, query: str) -> torch.Tensor:
        """
        クエリをトークンレベルで埋め込み
        
        Args:
            query: クエリテキスト
            
        Returns:
            トークンレベルの埋め込みテンソル (seq_len, hidden_dim)
        """
        return self.encode_document(query)  # 同じエンコーディング
    
    def build_index(self):
        """
        全文書の埋め込みを事前計算してインデックス構築
        メモリ効率化: 埋め込みをCPUに保存、定期的にキャッシュクリア
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        logger.info(f"Building ColBERT index for {len(self.documents)} documents (batched)...")

        texts = [d.content for d in self.documents]
        batch_size = 8
        # adaptively increase batch size if GPU memory available
        if self.device == 'cuda':
            batch_size = 16

        encoded_list = self.encode_documents_batch(texts, batch_size=batch_size)

        # Assign embeddings back to documents (already on CPU)
        for idx, emb in enumerate(encoded_list):
            if idx % 200 == 0:
                logger.info(f"  Assigned embeddings for doc {idx}/{len(self.documents)}")
            self.documents[idx].embeddings = emb

        # final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.index_built = True
        logger.info("✓ ColBERT index built successfully (batched)")
    
    def compute_colbert_score(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor
    ) -> float:
        """
        ColBERTスコア計算（MaxSim遅延相互作用）- GPU高速化
        
        各クエリトークンに対して、最も類似する文書トークンのスコアを合計
        
        Args:
            query_embeddings: クエリの埋め込み (q_len, hidden_dim)
            doc_embeddings: 文書の埋め込み (d_len, hidden_dim)
            
        Returns:
            ColBERTスコア
        """
        # GPUに移動して計算
        device = self.device if hasattr(self, 'device') else 'cpu'
        query_gpu = query_embeddings.to(device)
        doc_gpu = doc_embeddings.to(device)
        
        # コサイン類似度行列を計算 (q_len, d_len)
        query_norm = torch.nn.functional.normalize(query_gpu, p=2, dim=1)
        doc_norm = torch.nn.functional.normalize(doc_gpu, p=2, dim=1)
        
        similarity_matrix = torch.matmul(query_norm, doc_norm.T)  # (q_len, d_len)
        
        # 各クエリトークンの最大類似度を合計（MaxSim）
        max_similarities = similarity_matrix.max(dim=1)[0]  # (q_len,)
        
        # クエリトークン数で正規化（0-1のスケールに）
        num_query_tokens = query_gpu.size(0)
        colbert_score = max_similarities.sum().item() / num_query_tokens
        
        return colbert_score
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[str, float, Dict]]:
        """
        ColBERT検索（高速化: top_k*10候補に絞ってから詳細スコア計算）
        
        Args:
            query: 検索クエリ
            top_k: 返す結果数
            
        Returns:
            [(content, score, metadata), ...]
        """
        if not self.index_built:
            raise ValueError("Index not built. Call build_index() first.")
        
        # クエリ埋め込み
        query_embeddings = self.encode_query(query)
        
        # 高速化: 平均プーリングで候補を絞る（GPU使用）
        query_mean = query_embeddings.mean(dim=0, keepdim=True).to(self.device)  # (1, hidden_dim)
        
        candidate_count = min(top_k * 10, len(self.documents))  # 候補を10倍に絞る
        
        # 全文書の平均埋め込みをバッチ計算
        doc_means = torch.stack([doc.embeddings.mean(dim=0) for doc in self.documents]).to(self.device)
        
        # バッチでコサイン類似度計算
        simple_scores = torch.nn.functional.cosine_similarity(
            query_mean, doc_means, dim=1
        )
        
        # 上位候補のインデックスを取得
        top_indices = torch.topk(simple_scores, k=candidate_count).indices.cpu().numpy()
        
        # 詳細スコア計算（ColBERT MaxSim）- GPUで高速化
        scores = []
        for idx in top_indices:
            doc = self.documents[idx]
            score = self.compute_colbert_score(query_embeddings, doc.embeddings)
            scores.append((doc, score))
        
        # スコア順にソート
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Top-k取得
        results = []
        for doc, score in scores[:top_k]:
            results.append((
                doc.content,
                score,
                {**doc.metadata, 'doc_id': doc.doc_id}
            ))
        
        return results
    
    def save(self, filepath: Path):
        """
        インデックスを保存
        
        Args:
            filepath: 保存先ファイルパス
        """
        logger.info(f"Saving ColBERT index to {filepath}")
        
        save_data = {
            'documents': self.documents,
            'model_name': self.model_name,
            'index_built': self.index_built
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info("✓ Index saved")
    
    def load(self, filepath: Path):
        """
        インデックスを読み込み
        
        Args:
            filepath: 読み込み元ファイルパス
        """
        logger.info(f"Loading ColBERT index from {filepath}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.documents = save_data['documents']
        self.index_built = save_data['index_built']
        
        logger.info(f"✓ Loaded {len(self.documents)} documents")


def main():
    """テスト実行"""
    print("="*80)
    print("ColBERT RAG for Kasensabo")
    print("河川砂防ダムColBERT検索システム")
    print("="*80)
    
    # 初期化
    colbert = ColBERTRAG()
    
    # 文書読み込み
    num_docs = colbert.load_documents()
    print(f"\n✓ Loaded {num_docs} documents")
    
    # インデックス構築
    colbert.build_index()
    print("✓ Index built")
    
    # 保存
    save_path = OUTPUT_DIR / "colbert_rag.pkl"
    colbert.save(save_path)
    print(f"✓ Saved to {save_path}")
    
    # テスト検索
    test_queries = [
        "砂防ダムの点検で重視すべき項目は",
        "堤防の設計基準値は",
        "流量観測の手法"
    ]
    
    print("\n" + "="*80)
    print("Test Search")
    print("="*80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = colbert.search(query, top_k=3)
        
        for i, (content, score, metadata) in enumerate(results, 1):
            print(f"{i}. Score={score:.2f}, Source={metadata.get('source', 'N/A')}")
            print(f"   {content[:100]}...")


if __name__ == "__main__":
    main()
