"""
Benchmark System for ColBERT vs Naive RAG
ColBERTとNaive RAGの性能比較ベンチマーク
"""

import json
import time
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
import logging

from colbert_rag import ColBERTRAG
from config import *

logging.basicConfig(level=LOG_LEVEL, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class NaiveRAG:
    """
    比較用のNaive RAG実装（FAISS + Sentence-Transformers）
    raptor_mvpの実装を簡略化
    """
    
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import faiss
        
        logger.info("Initializing Naive RAG")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.documents = []
        self.index = None
        
    def load_documents(self):
        """文書読み込み"""
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        import glob as glob_module
        
        logger.info(f"Loading documents from {DATA_DIR}")
        
        # シンプルなテキストファイル読み込み
        raw_docs = []
        for filepath in DATA_DIR.glob("**/*.md"):
            try:
                loader = TextLoader(str(filepath), encoding='utf-8')
                raw_docs.extend(loader.load())
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(raw_docs)
        
        # サンプリング適用（ColBERTと同じ条件）
        if SAMPLE_RATIO < 1.0:
            import random
            sample_size = int(len(chunks) * SAMPLE_RATIO)
            chunks = random.sample(chunks, sample_size)
            logger.info(f"📊 Sampled {len(chunks)} chunks ({SAMPLE_RATIO*100:.0f}% of total)")
        
        self.documents = [(chunk.page_content, chunk.metadata) for chunk in chunks]
        logger.info(f"Loaded {len(self.documents)} documents")
        
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
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (Cosine similarity)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"✓ Index built: {len(self.documents)} vectors, dim={dimension}")
        
    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[str, float, Dict]]:
        """検索"""
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            content, metadata = self.documents[idx]
            results.append((content, float(score), metadata))
        
        return results


class BenchmarkSystem:
    """ベンチマークシステム"""
    
    def __init__(self):
        self.questions = []
        
    def load_questions(self, filepath: Path = BENCHMARK_QUESTIONS_FILE):
        """ベンチマーク質問を読み込み"""
        logger.info(f"Loading benchmark questions from {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # カテゴリ構造から質問を展開
            if 'categories' in data:
                question_id = 1
                for category in data['categories']:
                    category_name = category['name']
                    for question_text in category['questions']:
                        self.questions.append({
                            'id': question_id,
                            'category': category_name,
                            'question': question_text
                        })
                        question_id += 1
            else:
                self.questions = data.get('questions', [])
        
        logger.info(f"Loaded {len(self.questions)} questions")
        
    def compute_score(self, results: List[Tuple[str, float, Dict]]) -> Tuple[float, float]:
        """
        スコア計算
        
        Returns:
            (平均スコア, 最高スコア)
        """
        if not results:
            return 0.0, 0.0
        
        scores = [score for _, score, _ in results]
        return np.mean(scores), np.max(scores)
    
    def run_benchmark(self, rag_system, system_name: str) -> Dict:
        """
        ベンチマーク実行
        
        Args:
            rag_system: ColBERTRAGまたはNaiveRAG
            system_name: システム名
            
        Returns:
            ベンチマーク結果
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Running benchmark: {system_name}")
        logger.info(f"{'='*80}")
        
        results = []
        total_time = 0
        
        for idx, q in enumerate(self.questions, 1):
            if idx % 10 == 0:
                logger.info(f"Progress: {idx}/{len(self.questions)}")
            
            question = q['question']
            
            # 検索実行
            start_time = time.time()
            search_results = rag_system.search(question, top_k=TOP_K_RETRIEVAL)
            elapsed_time = time.time() - start_time
            
            total_time += elapsed_time
            
            # スコア計算
            avg_score, top_score = self.compute_score(search_results)
            
            results.append({
                'question_id': q['id'],
                'category': q['category'],
                'question': question,
                'time': elapsed_time,
                'avg_score': avg_score,
                'top_score': top_score
            })
        
        # 統計計算
        avg_time = total_time / len(self.questions) * 1000  # ms
        avg_scores = [r['avg_score'] for r in results]
        top_scores = [r['top_score'] for r in results]
        
        summary = {
            'system_name': system_name,
            'total_questions': len(self.questions),
            'avg_time_ms': avg_time,
            'avg_score': np.mean(avg_scores),
            'top_score': np.mean(top_scores),
            'score_std': np.std(avg_scores)
        }
        
        logger.info(f"\n{system_name} Results:")
        logger.info(f"  Average Time: {avg_time:.2f} ms")
        logger.info(f"  Average Score: {summary['avg_score']:.4f}")
        logger.info(f"  Average Top Score: {summary['top_score']:.4f}")
        
        return {
            'summary': summary,
            'details': results
        }


def main():
    """メイン実行"""
    print("="*80)
    print("ColBERT vs Naive RAG Benchmark")
    print("="*80)
    
    # ベンチマークシステム初期化
    benchmark = BenchmarkSystem()
    benchmark.load_questions()
    
    # Naive RAG構築
    print("\n" + "="*80)
    print("Building Naive RAG")
    print("="*80)
    naive_rag = NaiveRAG()
    naive_rag.load_documents()
    naive_rag.build_index()
    
    # ColBERT RAG構築
    print("\n" + "="*80)
    print("Building ColBERT RAG")
    print("="*80)
    colbert_rag = ColBERTRAG()
    colbert_rag.load_documents()
    colbert_rag.build_index()
    
    # ベンチマーク実行
    print("\n" + "="*80)
    print("Running Benchmarks")
    print("="*80)
    
    naive_results = benchmark.run_benchmark(naive_rag, "Naive RAG")
    colbert_results = benchmark.run_benchmark(colbert_rag, "ColBERT RAG")
    
    # 比較
    naive_score = naive_results['summary']['avg_score']
    colbert_score = colbert_results['summary']['avg_score']
    improvement = ((colbert_score - naive_score) / naive_score) * 100
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"Naive RAG Score:   {naive_score:.4f}")
    print(f"ColBERT RAG Score: {colbert_score:.4f}")
    print(f"Improvement:       {improvement:+.2f}%")
    
    # 結果保存
    final_results = {
        'naive_rag': naive_results,
        'colbert_rag': colbert_results,
        'comparison': {
            'naive_score': naive_score,
            'colbert_score': colbert_score,
            'improvement_pct': improvement
        }
    }
    
    output_path = BENCHMARK_RESULTS_FILE
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
