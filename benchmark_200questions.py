"""
200問の質問（専門性の粒度で分類）に対して
Naive RAG, RAPTOR, ColBERTを実行し、Selective RAGと比較
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# パス設定
DATA_DIR = Path("data/kasensabo_knowledge_base")
QUESTIONS_FILE = Path("selective_rag/output/questions_200.json")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

class UnifiedBenchmark:
    """200問に対する統一ベンチマーク"""
    
    def __init__(self):
        self.questions = self.load_questions()
        self.results = {
            "naive_rag": [],
            "raptor": [],
            "colbert_50pct": [],
            "selective_rag": []
        }
        # 一時的な質問ファイルパス
        self.temp_questions_file_raptor = Path("raptor_mvp/output/benchmark_questions_200.json")
        self.temp_questions_file_colbert = Path("colbert_mvp/output/benchmark_questions_200.json")
    
    def load_questions(self) -> List[Dict]:
        """200問の質問を読み込む"""
        with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✓ Loaded {len(data)} questions")
        return data
    
    def prepare_questions_for_raptor(self):
        """RAPTOR形式の質問ファイルを生成"""
        questions_raptor = []
        for i, q in enumerate(self.questions, 1):
            questions_raptor.append({
                "question_id": i,
                "category": q['category'],
                "question": q['question']
            })
        
        self.temp_questions_file_raptor.parent.mkdir(exist_ok=True)
        with open(self.temp_questions_file_raptor, 'w', encoding='utf-8') as f:
            json.dump(questions_raptor, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Created RAPTOR question file: {self.temp_questions_file_raptor}")
    
    def prepare_questions_for_colbert(self):
        """ColBERT形式の質問ファイルを生成"""
        questions_colbert = []
        for i, q in enumerate(self.questions, 1):
            questions_colbert.append({
                "question_id": i,
                "category": q['category'],
                "question": q['question']
            })
        
        self.temp_questions_file_colbert.parent.mkdir(exist_ok=True)
        with open(self.temp_questions_file_colbert, 'w', encoding='utf-8') as f:
            json.dump(questions_colbert, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Created ColBERT question file: {self.temp_questions_file_colbert}")
    
    def run_naive_rag(self):
        """Naive RAGで200問を評価"""
        logger.info("\n" + "="*80)
        logger.info("[1/4] Running Naive RAG on 200 questions")
        logger.info("="*80)
        
        # Naive RAG初期化
        rag = RaptorNaiveRAG(data_dir=str(DATA_DIR))
        rag.load_documents()
        rag.build_index()
        
        results = []
        for i, q_data in enumerate(self.questions, 1):
            question = q_data['question']
            
            start_time = time.time()
            search_results = rag.search(question, top_k=5)
            elapsed = (time.time() - start_time) * 1000  # ms
            
            scores = [r[1] for r in search_results]
            
            result = {
                "question_id": i,
                "question": question,
                "true_granularity": q_data['granularity'],
                "category": q_data['category'],
                "time_ms": elapsed,
                "avg_score": float(np.mean(scores)) if scores else 0.0,
                "top_score": float(max(scores)) if scores else 0.0,
                "num_results": len(search_results)
            }
            results.append(result)
            
            if i % 20 == 0:
                logger.info(f"  Progress: {i}/{len(self.questions)} questions")
        
        self.results["naive_rag"] = results
        logger.info(f"✓ Naive RAG completed: {len(results)} questions")
    
    def run_raptor(self):
        """RAPTORで200問を評価"""
        logger.info("\n" + "="*80)
        logger.info("[2/4] Running RAPTOR on 200 questions")
        logger.info("="*80)
        
        # RAPTOR初期化
        rag = RAPTORRAG(data_dir=str(DATA_DIR))
        
        # 既存のツリーがあれば読み込み、なければ構築
        raptor_pkl = Path("raptor_mvp/output/raptor_rag.pkl")
        if raptor_pkl.exists():
            logger.info("Loading existing RAPTOR tree...")
            rag.load()
        else:
            logger.info("Building RAPTOR tree (this may take a while)...")
            rag.load_documents()
            rag.build_tree()
            rag.save()
        
        results = []
        for i, q_data in enumerate(self.questions, 1):
            question = q_data['question']
            
            start_time = time.time()
            search_results = rag.search(question, top_k=5)
            elapsed = (time.time() - start_time) * 1000  # ms
            
            scores = [r[1] for r in search_results]
            
            result = {
                "question_id": i,
                "question": question,
                "true_granularity": q_data['granularity'],
                "category": q_data['category'],
                "time_ms": elapsed,
                "avg_score": float(np.mean(scores)) if scores else 0.0,
                "top_score": float(max(scores)) if scores else 0.0,
                "num_results": len(search_results)
            }
            results.append(result)
            
            if i % 20 == 0:
                logger.info(f"  Progress: {i}/{len(self.questions)} questions")
        
        self.results["raptor"] = results
        logger.info(f"✓ RAPTOR completed: {len(results)} questions")
    
    def run_colbert(self):
        """ColBERT (50% sampling)で200問を評価"""
        logger.info("\n" + "="*80)
        logger.info("[3/4] Running ColBERT (50% sampling) on 200 questions")
        logger.info("="*80)
        
        # ColBERT初期化
        rag = ColBERTRAG(data_dir=str(DATA_DIR), sample_ratio=0.5)
        rag.load_documents()
        rag.build_index()
        
        results = []
        for i, q_data in enumerate(self.questions, 1):
            question = q_data['question']
            
            start_time = time.time()
            search_results = rag.search(question, top_k=5)
            elapsed = (time.time() - start_time) * 1000  # ms
            
            scores = [r[1] for r in search_results]
            
            result = {
                "question_id": i,
                "question": question,
                "true_granularity": q_data['granularity'],
                "category": q_data['category'],
                "time_ms": elapsed,
                "avg_score": float(np.mean(scores)) if scores else 0.0,
                "top_score": float(max(scores)) if scores else 0.0,
                "num_results": len(search_results)
            }
            results.append(result)
            
            if i % 20 == 0:
                logger.info(f"  Progress: {i}/{len(self.questions)} questions")
        
        self.results["colbert_50pct"] = results
        logger.info(f"✓ ColBERT completed: {len(results)} questions")
    
    def load_selective_results(self):
        """既存のSelective RAG結果を読み込む"""
        logger.info("\n" + "="*80)
        logger.info("[4/4] Loading Selective RAG results")
        logger.info("="*80)
        
        selective_file = Path("selective_rag/output/selective_rag_benchmark_results.json")
        with open(selective_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.results["selective_rag"] = data['results']
        logger.info(f"✓ Loaded {len(data['results'])} Selective RAG results")
    
    def compute_statistics(self):
        """統計情報を計算"""
        logger.info("\n" + "="*80)
        logger.info("Computing Statistics")
        logger.info("="*80)
        
        stats = {}
        
        for system_name, results in self.results.items():
            if not results:
                continue
            
            # スコア抽出
            avg_scores = [r['avg_score'] for r in results]
            top_scores = [r['top_score'] for r in results]
            times = [r['time_ms'] for r in results]
            
            # 粒度別スコア
            fine_scores = [r['avg_score'] for r in results if r['true_granularity'] == 'fine']
            coarse_scores = [r['avg_score'] for r in results if r['true_granularity'] == 'coarse']
            
            stats[system_name] = {
                "total_questions": len(results),
                "avg_score": float(np.mean(avg_scores)),
                "top_score": float(np.mean(top_scores)),
                "score_std": float(np.std(avg_scores)),
                "avg_time_ms": float(np.mean(times)),
                "time_std_ms": float(np.std(times)),
                "fine_grained": {
                    "count": len(fine_scores),
                    "avg_score": float(np.mean(fine_scores)) if fine_scores else 0.0,
                    "score_std": float(np.std(fine_scores)) if fine_scores else 0.0
                },
                "coarse_grained": {
                    "count": len(coarse_scores),
                    "avg_score": float(np.mean(coarse_scores)) if coarse_scores else 0.0,
                    "score_std": float(np.std(coarse_scores)) if coarse_scores else 0.0
                }
            }
        
        # Selective RAGの選択精度
        if "selective_rag" in self.results:
            correct = sum(1 for r in self.results["selective_rag"] if r.get('correct_selection', False))
            stats["selective_rag"]["selection_accuracy"] = correct / len(self.results["selective_rag"])
            
            naive_selected = sum(1 for r in self.results["selective_rag"] if r['selected_system'] == 'naive')
            colbert_selected = sum(1 for r in self.results["selective_rag"] if r['selected_system'] == 'colbert')
            
            stats["selective_rag"]["system_usage"] = {
                "naive": naive_selected,
                "colbert": colbert_selected
            }
        
        return stats
    
    def save_results(self, stats):
        """結果を保存"""
        output = {
            "metadata": {
                "total_questions": len(self.questions),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "statistics": stats,
            "detailed_results": self.results
        }
        
        output_file = OUTPUT_DIR / "unified_benchmark_200q_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✓ Results saved to {output_file}")
        return output_file
    
    def print_comparison(self, stats):
        """比較結果を表示"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS - 200 Questions Comparison")
        print("="*80)
        
        systems = ["naive_rag", "raptor", "colbert_50pct", "selective_rag"]
        names = {
            "naive_rag": "Naive RAG",
            "raptor": "RAPTOR",
            "colbert_50pct": "ColBERT (50%)",
            "selective_rag": "Selective RAG"
        }
        
        # 総合スコア
        print("\n【Overall Performance】")
        print(f"{'System':<20} {'Avg Score':>12} {'Top Score':>12} {'Avg Time(ms)':>15} {'Std Dev':>10}")
        print("-"*80)
        
        baseline_score = stats["naive_rag"]["avg_score"]
        baseline_time = stats["naive_rag"]["avg_time_ms"]
        
        for sys in systems:
            if sys not in stats:
                continue
            s = stats[sys]
            score_diff = ((s["avg_score"] - baseline_score) / baseline_score * 100)
            time_ratio = s["avg_time_ms"] / baseline_time
            
            print(f"{names[sys]:<20} {s['avg_score']:>12.4f} {s['top_score']:>12.4f} "
                  f"{s['avg_time_ms']:>12.1f} ({time_ratio:>4.1f}x) {s['score_std']:>10.4f}")
            if sys != "naive_rag":
                print(f"{'':>20} {score_diff:>11.1f}%")
        
        # 粒度別スコア
        print("\n【Performance by Granularity】")
        print(f"{'System':<20} {'Fine-grained':>15} {'Coarse-grained':>17} {'Difference':>12}")
        print("-"*80)
        
        for sys in systems:
            if sys not in stats:
                continue
            s = stats[sys]
            fine = s["fine_grained"]["avg_score"]
            coarse = s["coarse_grained"]["avg_score"]
            diff = fine - coarse
            
            print(f"{names[sys]:<20} {fine:>15.4f} {coarse:>17.4f} {diff:>12.4f}")
        
        # Selective RAG特有の情報
        if "selective_rag" in stats:
            print("\n【Selective RAG Details】")
            sel = stats["selective_rag"]
            print(f"  Selection Accuracy: {sel['selection_accuracy']*100:.1f}%")
            print(f"  System Usage:")
            print(f"    - Naive RAG:  {sel['system_usage']['naive']} questions ({sel['system_usage']['naive']/200*100:.1f}%)")
            print(f"    - ColBERT:    {sel['system_usage']['colbert']} questions ({sel['system_usage']['colbert']/200*100:.1f}%)")
        
        print("\n" + "="*80)
        print("Key Findings:")
        print("="*80)
        
        # ベストシステム判定
        best_accuracy = max(stats[sys]["avg_score"] for sys in systems if sys in stats)
        fastest = min(stats[sys]["avg_time_ms"] for sys in systems if sys in stats)
        
        best_sys = [names[sys] for sys in systems if sys in stats and stats[sys]["avg_score"] == best_accuracy][0]
        fastest_sys = [names[sys] for sys in systems if sys in stats and stats[sys]["avg_time_ms"] == fastest][0]
        
        print(f"\n✓ Best Accuracy: {best_sys} ({best_accuracy:.4f})")
        print(f"✓ Fastest System: {fastest_sys} ({fastest:.1f}ms)")
        
        # 細粒度と粗粒度での優劣
        fine_best = max(stats[sys]["fine_grained"]["avg_score"] for sys in systems if sys in stats)
        coarse_best = max(stats[sys]["coarse_grained"]["avg_score"] for sys in systems if sys in stats)
        
        fine_sys = [names[sys] for sys in systems if sys in stats and stats[sys]["fine_grained"]["avg_score"] == fine_best][0]
        coarse_sys = [names[sys] for sys in systems if sys in stats and stats[sys]["coarse_grained"]["avg_score"] == coarse_best][0]
        
        print(f"\n✓ Best for Fine-grained: {fine_sys} ({fine_best:.4f})")
        print(f"✓ Best for Coarse-grained: {coarse_sys} ({coarse_best:.4f})")
        
        print("\n")

def main():
    """メイン処理"""
    print("="*80)
    print("Unified Benchmark: 200 Questions Evaluation")
    print("="*80)
    print("\nThis benchmark evaluates 4 systems on the same 200 questions:")
    print("  1. Naive RAG")
    print("  2. RAPTOR")
    print("  3. ColBERT (50% sampling)")
    print("  4. Selective RAG (adaptive)")
    print("\n")
    
    benchmark = UnifiedBenchmark()
    
    # 各システムで実行
    benchmark.run_naive_rag()
    benchmark.run_raptor()
    benchmark.run_colbert()
    benchmark.load_selective_results()
    
    # 統計計算
    stats = benchmark.compute_statistics()
    
    # 結果保存
    benchmark.save_results(stats)
    
    # 比較表示
    benchmark.print_comparison(stats)
    
    print("\n✓ Benchmark Complete!")

if __name__ == "__main__":
    main()
