"""
Benchmark: Naive RAG vs RAPTOR
ベンチマーク比較実装
"""
import json
import time
from pathlib import Path
from typing import Dict, List
import logging
from tqdm import tqdm

from config import *
from naive_rag import NaiveRAG
from raptor_rag import RAPTORRAG

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class RAGBenchmark:
    """RAG性能ベンチマーク"""
    
    def __init__(self):
        self.naive_rag = None
        self.raptor_rag = None
        self.questions = []
        self.results = {
            'naive': [],
            'raptor': [],
            'comparison': {}
        }
    
    def load_questions(self, questions_file: Path) -> None:
        """質問データを読み込み"""
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.questions = []
        for category in data['categories']:
            for question in category['questions']:
                self.questions.append({
                    'category': category['name'],
                    'question': question
                })
        
        logger.info(f"Loaded {len(self.questions)} questions")
    
    def setup_rags(self) -> None:
        """RAGシステムをセットアップ"""
        logger.info("Setting up RAG systems...")
        
        # Naive RAG
        logger.info("\n=== Naive RAG Setup ===")
        self.naive_rag = NaiveRAG()
        
        naive_cache = OUTPUT_DIR / "naive_rag.pkl"
        if naive_cache.exists():
            logger.info("Loading from cache...")
            self.naive_rag.load(naive_cache)
        else:
            self.naive_rag.load_documents(DATA_DIR)
            self.naive_rag.build_index()
            self.naive_rag.save(naive_cache)
        
        # RAPTOR RAG
        logger.info("\n=== RAPTOR RAG Setup ===")
        self.raptor_rag = RAPTORRAG()
        
        raptor_cache = OUTPUT_DIR / "raptor_rag.pkl"
        if raptor_cache.exists():
            logger.info("Loading from cache...")
            self.raptor_rag.load(raptor_cache)
        else:
            self.raptor_rag.load_documents(DATA_DIR)
            self.raptor_rag.build_tree()
            self.raptor_rag.build_index()
            self.raptor_rag.save(raptor_cache)
        
        logger.info("\nSetup complete!")
    
    def run_benchmark(self, sample_size: int = None) -> None:
        """ベンチマークを実行"""
        questions_to_test = self.questions[:sample_size] if sample_size else self.questions
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Running benchmark on {len(questions_to_test)} questions")
        logger.info(f"{'='*80}\n")
        
        for idx, item in enumerate(tqdm(questions_to_test, desc="Benchmarking"), 1):
            question = item['question']
            category = item['category']
            
            # Naive RAG
            naive_start = time.time()
            naive_results = self.naive_rag.search(question, top_k=TOP_K_RETRIEVAL)
            naive_time = time.time() - naive_start
            
            # RAPTOR RAG
            raptor_start = time.time()
            raptor_results = self.raptor_rag.search(question, top_k=TOP_K_RETRIEVAL)
            raptor_time = time.time() - raptor_start
            
            # 結果を記録
            self.results['naive'].append({
                'question_id': idx,
                'category': category,
                'question': question,
                'time': naive_time,
                'num_results': len(naive_results),
                'top_score': naive_results[0][1] if naive_results else 0,
                'avg_score': sum(r[1] for r in naive_results) / len(naive_results) if naive_results else 0,
                'sources': [r[2].get('source', 'N/A') for r in naive_results]
            })
            
            self.results['raptor'].append({
                'question_id': idx,
                'category': category,
                'question': question,
                'time': raptor_time,
                'num_results': len(raptor_results),
                'top_score': raptor_results[0][1] if raptor_results else 0,
                'avg_score': sum(r[1] for r in raptor_results) / len(raptor_results) if raptor_results else 0,
                'sources': [r[2].get('source', 'N/A') for r in raptor_results],
                'levels': [r[2].get('level', 'N/A') for r in raptor_results]
            })
        
        # 統計計算
        self._compute_statistics()
    
    def _compute_statistics(self) -> None:
        """統計を計算"""
        naive_results = self.results['naive']
        raptor_results = self.results['raptor']
        
        # 平均時間
        naive_avg_time = sum(r['time'] for r in naive_results) / len(naive_results)
        raptor_avg_time = sum(r['time'] for r in raptor_results) / len(raptor_results)
        
        # 平均スコア
        naive_avg_score = sum(r['avg_score'] for r in naive_results) / len(naive_results)
        raptor_avg_score = sum(r['avg_score'] for r in raptor_results) / len(raptor_results)
        
        # トップスコア平均
        naive_top_score = sum(r['top_score'] for r in naive_results) / len(naive_results)
        raptor_top_score = sum(r['top_score'] for r in raptor_results) / len(raptor_results)
        
        self.results['comparison'] = {
            'total_questions': len(naive_results),
            'naive': {
                'avg_time': naive_avg_time,
                'avg_score': naive_avg_score,
                'top_score': naive_top_score
            },
            'raptor': {
                'avg_time': raptor_avg_time,
                'avg_score': raptor_avg_score,
                'top_score': raptor_top_score
            },
            'improvement': {
                'time_ratio': raptor_avg_time / naive_avg_time if naive_avg_time > 0 else 0,
                'score_improvement': ((raptor_avg_score - naive_avg_score) / naive_avg_score * 100) if naive_avg_score > 0 else 0,
                'top_score_improvement': ((raptor_top_score - naive_top_score) / naive_top_score * 100) if naive_top_score > 0 else 0
            }
        }
    
    def print_summary(self) -> None:
        """結果サマリーを表示"""
        comp = self.results['comparison']
        
        print(f"\n{'='*80}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"Total Questions: {comp['total_questions']}\n")
        
        print("Naive RAG:")
        print(f"  Average Time:      {comp['naive']['avg_time']*1000:.2f} ms")
        print(f"  Average Score:     {comp['naive']['avg_score']:.4f}")
        print(f"  Top Score:         {comp['naive']['top_score']:.4f}\n")
        
        print("RAPTOR RAG:")
        print(f"  Average Time:      {comp['raptor']['avg_time']*1000:.2f} ms")
        print(f"  Average Score:     {comp['raptor']['avg_score']:.4f}")
        print(f"  Top Score:         {comp['raptor']['top_score']:.4f}\n")
        
        print("Improvement:")
        print(f"  Time Ratio:        {comp['improvement']['time_ratio']:.2f}x")
        print(f"  Score Improvement: {comp['improvement']['score_improvement']:+.2f}%")
        print(f"  Top Score Improvement: {comp['improvement']['top_score_improvement']:+.2f}%")
        
        print(f"\n{'='*80}\n")
    
    def save_results(self, output_file: Path) -> None:
        """結果を保存"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    benchmark = RAGBenchmark()
    
    # 質問読み込み
    benchmark.load_questions(BENCHMARK_QUESTIONS_FILE)
    
    # RAGセットアップ
    benchmark.setup_rags()
    
    # ベンチマーク実行（サンプルとして最初の10問）
    benchmark.run_benchmark(sample_size=10)
    
    # 結果表示
    benchmark.print_summary()
    
    # 結果保存
    benchmark.save_results(BENCHMARK_RESULTS_FILE)
    
    logger.info("\nBenchmark complete!")
