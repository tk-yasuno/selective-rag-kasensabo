"""
ColBERT MVP - 200問ベンチマーク
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from colbert_rag import ColBERTRAG
from sentence_transformers import SentenceTransformer
import faiss
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUESTIONS_FILE = Path("output/benchmark_questions_200.json")
OUTPUT_FILE = Path("output/colbert_benchmark_results_200q.json")
DATA_DIR = Path("../data/kasensabo_knowledge_base")

class NaiveRAG:
    """比較用のNaive RAG実装"""
    
    def __init__(self):
        logger.info("Initializing Naive RAG")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.documents = []
        self.index = None
    
    def load_documents(self):
        """文書読み込み"""
        logger.info(f"Loading documents from {DATA_DIR}")
        
        raw_docs = []
        for filepath in DATA_DIR.glob("**/*.md"):
            try:
                loader = TextLoader(str(filepath), encoding='utf-8')
                raw_docs.extend(loader.load())
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        # チャンク分割
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )
        
        self.documents = []
        for doc in raw_docs:
            chunks = splitter.split_text(doc.page_content)
            for chunk in chunks:
                self.documents.append({
                    'content': chunk,
                    'source': doc.metadata.get('source', 'unknown')
                })
        
        logger.info(f"✓ Loaded {len(self.documents)} document chunks")
    
    def build_index(self):
        """FAISSインデックス構築"""
        logger.info("Building FAISS index...")
        
        contents = [doc['content'] for doc in self.documents]
        embeddings = self.embedding_model.encode(
            contents,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        # FAISS IndexFlatIP (内積)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"✓ Built FAISS index with {len(self.documents)} documents")
    
    def search(self, query: str, top_k: int = 5):
        """検索実行"""
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        ).astype('float32')
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append((
                self.documents[idx]['content'],
                float(score),
                self.documents[idx]
            ))
        
        return results

def load_questions():
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_naive_rag(questions):
    """Naive RAGで200問評価"""
    logger.info("Running Naive RAG on 200 questions...")
    rag = NaiveRAG()
    rag.load_documents()
    rag.build_index()
    
    results = []
    for i, q in enumerate(questions, 1):
        start = time.time()
        search_results = rag.search(q['question'], top_k=5)
        elapsed = (time.time() - start) * 1000  # ms
        
        scores = [r[1] for r in search_results]
        results.append({
            "question_id": q['question_id'],
            "category": q['category'],
            "question": q['question'],
            "granularity": q.get('granularity', 'unknown'),
            "time": elapsed,
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "top_score": float(max(scores)) if scores else 0.0
        })
        
        if i % 20 == 0:
            logger.info(f"  Progress: {i}/{len(questions)}")
    
    return results

def run_colbert(questions):
    """ColBERT (50% sampling)で200問評価"""
    logger.info("Running ColBERT (50%) on 200 questions...")
    # Note: ColBERTRAGはconfig.pyのSAMPLE_RATIOを使用 (デフォルト0.5)
    rag = ColBERTRAG()
    rag.load_documents()
    rag.build_index()
    
    results = []
    for i, q in enumerate(questions, 1):
        start = time.time()
        search_results = rag.search(q['question'], top_k=5)
        elapsed = (time.time() - start) * 1000  # ms
        
        scores = [r[1] for r in search_results]
        results.append({
            "question_id": q['question_id'],
            "category": q['category'],
            "question": q['question'],
            "granularity": q.get('granularity', 'unknown'),
            "time": elapsed,
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "top_score": float(max(scores)) if scores else 0.0
        })
        
        if i % 20 == 0:
            logger.info(f"  Progress: {i}/{len(questions)}")
    
    return results

def compute_summary(results):
    """統計情報を計算"""
    avg_scores = [r['avg_score'] for r in results]
    top_scores = [r['top_score'] for r in results]
    times = [r['time'] for r in results]
    
    fine_scores = [r['avg_score'] for r in results if r.get('granularity') == 'fine']
    coarse_scores = [r['avg_score'] for r in results if r.get('granularity') == 'coarse']
    
    return {
        "total_questions": len(results),
        "avg_time_ms": float(np.mean(times)),
        "avg_score": float(np.mean(avg_scores)),
        "top_score": float(np.mean(top_scores)),
        "score_std": float(np.std(avg_scores)),
        "fine_grained_score": float(np.mean(fine_scores)) if fine_scores else 0.0,
        "coarse_grained_score": float(np.mean(coarse_scores)) if coarse_scores else 0.0
    }

def main():
    questions = load_questions()
    logger.info(f"Loaded {len(questions)} questions")
    
    naive_results = run_naive_rag(questions)
    colbert_results = run_colbert(questions)
    
    output = {
        "naive_rag": {
            "summary": compute_summary(naive_results),
            "details": naive_results
        },
        "colbert_rag": {
            "summary": compute_summary(colbert_results),
            "details": colbert_results
        }
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ Results saved to {OUTPUT_FILE}")
    
    # サマリー表示
    print("\n" + "="*80)
    print("Benchmark Results Summary (200 questions)")
    print("="*80)
    print(f"Naive RAG:")
    print(f"  Avg Score: {output['naive_rag']['summary']['avg_score']:.4f}")
    print(f"  Avg Time:  {output['naive_rag']['summary']['avg_time_ms']:.1f}ms")
    print(f"ColBERT RAG (50%):")
    print(f"  Avg Score: {output['colbert_rag']['summary']['avg_score']:.4f}")
    print(f"  Avg Time:  {output['colbert_rag']['summary']['avg_time_ms']:.1f}ms")

if __name__ == "__main__":
    main()
