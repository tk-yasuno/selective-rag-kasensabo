"""
Selective RAG Benchmark System
200問でNaive RAG、ColBERT RAG、Selective RAGを比較評価
"""

import json
import time
import logging
from typing import Dict, List
from pathlib import Path

from config import *
from naive_rag import NaiveRAG
from colbert_rag_module import ColBERTRAG
from selective_agent import SelectiveAgent

logger = logging.getLogger(__name__)


class SelectiveRAGSystem:
    """
    Selective RAG: 質問粒度に応じて最適なRAGシステムを選択
    """
    
    def __init__(self):
        """システム初期化"""
        logger.info("=" * 60)
        logger.info("Initializing Selective RAG System")
        logger.info("=" * 60)
        
        # エージェント初期化
        self.agent = SelectiveAgent()
        
        # Naive RAG初期化
        logger.info("\n[1/2] Building Naive RAG...")
        self.naive_rag = NaiveRAG(model_name=NAIVE_MODEL)
        self.naive_rag.load_documents(
            data_dir=DATA_DIR,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            sample_ratio=1.0  # 全データ
        )
        self.naive_rag.build_index()
        
        # ColBERT RAG初期化
        logger.info("\n[2/2] Building ColBERT RAG...")
        self.colbert_rag = ColBERTRAG(model_name=COLBERT_MODEL)
        self.colbert_rag.load_documents(
            data_dir=DATA_DIR,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            sample_ratio=COLBERT_SAMPLE_RATIO  # 50%サンプリング
        )
        self.colbert_rag.build_index(batch_size=16)
        
        logger.info("\n✓ Selective RAG System Ready")
        
        # 統計
        self.stats = {
            "naive_selections": 0,
            "colbert_selections": 0,
            "total_queries": 0,
        }
    
    def query(self, question: str, top_k: int = TOP_K_RETRIEVAL) -> Dict:
        """
        Selective RAG実行
        
        Args:
            question: 質問文
            top_k: 取得件数
        
        Returns:
            結果辞書
        """
        # エージェントによる選択
        selection = self.agent.select_rag_system(question)
        selected_system = selection["selected_system"]
        
        # 選択されたRAGで検索
        start_time = time.time()
        
        if selected_system == "naive":
            results = self.naive_rag.search(question, top_k=top_k)
            self.stats["naive_selections"] += 1
        else:  # colbert
            results = self.colbert_rag.search(question, top_k=top_k)
            self.stats["colbert_selections"] += 1
        
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats["total_queries"] += 1
        
        return {
            "question": question,
            "selected_system": selected_system,
            "granularity": selection["granularity"],
            "results": results,
            "time_ms": elapsed_ms,
        }
    
    def get_stats(self) -> Dict:
        """統計情報取得"""
        total = self.stats["total_queries"]
        if total > 0:
            naive_pct = (self.stats["naive_selections"] / total) * 100
            colbert_pct = (self.stats["colbert_selections"] / total) * 100
        else:
            naive_pct = colbert_pct = 0
        
        return {
            **self.stats,
            "naive_percentage": naive_pct,
            "colbert_percentage": colbert_pct,
        }


class BenchmarkSystem:
    """ベンチマークシステム"""
    
    def __init__(self):
        """初期化"""
        self.questions = []
    
    def load_questions(self, questions_file: Path = QUESTIONS_FILE):
        """質問データセット読み込み"""
        logger.info(f"Loading questions from {questions_file}")
        
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.questions = data["questions"]
        logger.info(f"✓ Loaded {len(self.questions)} questions")
        logger.info(f"  - Fine: {data['metadata']['fine_grained']}")
        logger.info(f"  - Coarse: {data['metadata']['coarse_grained']}")
    
    def compute_score(self, results: List[tuple]) -> tuple:
        """
        スコア計算（平均スコア、トップスコア）
        
        Args:
            results: [(content, score, metadata), ...]
        
        Returns:
            (avg_score, top_score)
        """
        if not results:
            return 0.0, 0.0
        
        scores = [score for _, score, _ in results]
        avg_score = sum(scores) / len(scores)
        top_score = max(scores)
        
        return avg_score, top_score
    
    def run_benchmark(self):
        """ベンチマーク実行"""
        logger.info("\n" + "=" * 60)
        logger.info("Running Benchmark on 200 Questions")
        logger.info("=" * 60)
        
        # システム初期化
        selective_system = SelectiveRAGSystem()
        
        # 結果格納
        results = {
            "metadata": {
                "total_questions": len(self.questions),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "results": [],
        }
        
        # 各質問で評価
        for i, q_data in enumerate(self.questions, 1):
            question = q_data["question"]
            true_granularity = q_data["granularity"]
            
            # Selective RAG実行
            response = selective_system.query(question)
            
            # スコア計算
            avg_score, top_score = self.compute_score(response["results"])
            
            result = {
                "question_id": q_data["id"],
                "question": question,
                "true_granularity": true_granularity,
                "predicted_granularity": response["granularity"],
                "selected_system": response["selected_system"],
                "time_ms": response["time_ms"],
                "avg_score": avg_score,
                "top_score": top_score,
                "correct_selection": (
                    (true_granularity == "fine" and response["selected_system"] == "colbert") or
                    (true_granularity == "coarse" and response["selected_system"] == "naive")
                ),
            }
            
            results["results"].append(result)
            
            # 進捗表示
            if i % 20 == 0:
                logger.info(f"  Progress: {i}/{len(self.questions)} questions")
        
        # 統計計算
        stats = selective_system.get_stats()
        
        # 選択精度
        correct_selections = sum(1 for r in results["results"] if r["correct_selection"])
        selection_accuracy = (correct_selections / len(self.questions)) * 100
        
        # 平均スコア
        avg_scores = [r["avg_score"] for r in results["results"]]
        overall_avg_score = sum(avg_scores) / len(avg_scores)
        
        # 平均時間
        times = [r["time_ms"] for r in results["results"]]
        overall_avg_time = sum(times) / len(times)
        
        # 粒度別スコア
        fine_results = [r for r in results["results"] if r["true_granularity"] == "fine"]
        coarse_results = [r for r in results["results"] if r["true_granularity"] == "coarse"]
        
        fine_avg_score = sum(r["avg_score"] for r in fine_results) / len(fine_results) if fine_results else 0
        coarse_avg_score = sum(r["avg_score"] for r in coarse_results) / len(coarse_results) if coarse_results else 0
        
        results["summary"] = {
            "system_name": "Selective RAG",
            "total_questions": len(self.questions),
            "selection_accuracy": selection_accuracy,
            "correct_selections": correct_selections,
            "naive_selections": stats["naive_selections"],
            "colbert_selections": stats["colbert_selections"],
            "naive_percentage": stats["naive_percentage"],
            "colbert_percentage": stats["colbert_percentage"],
            "overall_avg_score": overall_avg_score,
            "overall_avg_time_ms": overall_avg_time,
            "fine_grained_avg_score": fine_avg_score,
            "coarse_grained_avg_score": coarse_avg_score,
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)
        logger.info(f"Selection Accuracy: {selection_accuracy:.2f}%")
        logger.info(f"  - Correct: {correct_selections}/{len(self.questions)}")
        logger.info(f"\nSystem Usage:")
        logger.info(f"  - Naive RAG: {stats['naive_selections']} ({stats['naive_percentage']:.1f}%)")
        logger.info(f"  - ColBERT RAG: {stats['colbert_selections']} ({stats['colbert_percentage']:.1f}%)")
        logger.info(f"\nPerformance:")
        logger.info(f"  - Overall Avg Score: {overall_avg_score:.4f}")
        logger.info(f"  - Fine-grained Score: {fine_avg_score:.4f}")
        logger.info(f"  - Coarse-grained Score: {coarse_avg_score:.4f}")
        logger.info(f"  - Avg Time: {overall_avg_time:.2f} ms")
        logger.info("=" * 60)
        
        return results
    
    def save_results(self, results: Dict, output_file: Path = BENCHMARK_OUTPUT_FILE):
        """結果保存"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✓ Results saved to {output_file}")


def main():
    """メイン実行"""
    # ベンチマーク実行
    benchmark = BenchmarkSystem()
    benchmark.load_questions()
    
    results = benchmark.run_benchmark()
    benchmark.save_results(results)
    
    print("\n✓ Benchmark Complete!")
    print(f"Results: {BENCHMARK_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
