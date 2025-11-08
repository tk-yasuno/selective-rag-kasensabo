"""
200問の質問に対してRAPTOR MVPとColBERT MVPを実行
（各MVPのbenchmark.pyを修正して200問対応）
"""

import json
import subprocess
import shutil
from pathlib import Path

# パス設定
QUESTIONS_200 = Path("selective_rag/output/questions_200.json")
RAPTOR_QUESTIONS = Path("raptor_mvp/output/benchmark_questions_200.json")
COLBERT_QUESTIONS = Path("colbert_mvp/output/benchmark_questions_200.json")

def prepare_question_files():
    """200問を各MVPの形式に変換"""
    print("="*80)
    print("Preparing 200-question files for RAPTOR and ColBERT MVPs")
    print("="*80)
    
    # 元の質問を読み込み
    with open(QUESTIONS_200, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions_data = data['questions']  # questionsキーから取得
    
    # RAPTOR/ColBERT形式に変換
    questions_list = []
    for q in questions_data:
        questions_list.append({
            "question_id": q['id'],
            "category": q['category'],
            "question": q['question'],
            "granularity": q['granularity']  # 粒度情報も保持
        })
    
    # RAPTOR用に保存
    RAPTOR_QUESTIONS.parent.mkdir(exist_ok=True)
    with open(RAPTOR_QUESTIONS, 'w', encoding='utf-8') as f:
        json.dump(questions_list, f, ensure_ascii=False, indent=2)
    print(f"✓ Created: {RAPTOR_QUESTIONS}")
    
    # ColBERT用に保存
    COLBERT_QUESTIONS.parent.mkdir(exist_ok=True)
    with open(COLBERT_QUESTIONS, 'w', encoding='utf-8') as f:
        json.dump(questions_list, f, ensure_ascii=False, indent=2)
    print(f"✓ Created: {COLBERT_QUESTIONS}")
    
    print(f"\n✓ Total questions: {len(questions_list)}")
    print(f"  - Fine-grained: {sum(1 for q in questions_list if q['granularity'] == 'fine')}")
    print(f"  - Coarse-grained: {sum(1 for q in questions_list if q['granularity'] == 'coarse')}")
    print()

def check_benchmark_scripts():
    """benchmark.pyの存在確認"""
    raptor_bench = Path("raptor_mvp/benchmark_200q.py")
    colbert_bench = Path("colbert_mvp/benchmark_200q.py")
    
    print("="*80)
    print("Next Steps - Manual Execution Required")
    print("="*80)
    print("""
To run benchmarks on 200 questions, you need to:

1. RAPTOR MVP (200 questions):
   cd raptor_mvp
   python benchmark_200q.py
   
   Or modify benchmark.py to use benchmark_questions_200.json

2. ColBERT MVP (200 questions):
   cd colbert_mvp
   python benchmark_200q.py
   
   Or modify benchmark.py to use benchmark_questions_200.json

3. Results will be saved as:
   - raptor_mvp/output/benchmark_results_200q.json
   - colbert_mvp/output/colbert_benchmark_results_200q.json

4. Then run comparison:
   python compare_all_200q.py
""")

def create_raptor_benchmark_script():
    """RAPTOR用の200問ベンチマークスクリプトを作成"""
    raptor_bench_content = '''"""
RAPTOR MVP - 200問ベンチマーク
"""

import json
import time
import logging
from pathlib import Path
from naive_rag import NaiveRAG
from raptor_rag import RAPTORRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUESTIONS_FILE = Path("output/benchmark_questions_200.json")
OUTPUT_FILE = Path("output/benchmark_results_200q.json")

def load_questions():
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_naive_rag(questions):
    """Naive RAGで200問評価"""
    logger.info("Running Naive RAG on 200 questions...")
    rag = NaiveRAG(data_dir="../data/kasensabo_knowledge_base")
    rag.load_documents()
    rag.build_index()
    
    results = []
    for i, q in enumerate(questions, 1):
        start = time.time()
        search_results = rag.search(q['question'], top_k=5)
        elapsed = time.time() - start
        
        scores = [r[1] for r in search_results]
        results.append({
            "question_id": q['question_id'],
            "category": q['category'],
            "question": q['question'],
            "granularity": q.get('granularity', 'unknown'),
            "time": elapsed,
            "num_results": len(search_results),
            "top_score": float(max(scores)) if scores else 0.0,
            "avg_score": float(sum(scores)/len(scores)) if scores else 0.0,
            "sources": [r[2]['source'] for r in search_results]
        })
        
        if i % 20 == 0:
            logger.info(f"  Progress: {i}/{len(questions)}")
    
    return results

def run_raptor(questions):
    """RAPTORで200問評価"""
    logger.info("Running RAPTOR on 200 questions...")
    rag = RAPTORRAG(data_dir="../data/kasensabo_knowledge_base")
    
    raptor_pkl = Path("output/raptor_rag.pkl")
    if raptor_pkl.exists():
        logger.info("Loading existing RAPTOR tree...")
        rag.load()
    else:
        logger.info("Building RAPTOR tree...")
        rag.load_documents()
        rag.build_tree()
        rag.save()
    
    results = []
    for i, q in enumerate(questions, 1):
        start = time.time()
        search_results = rag.search(q['question'], top_k=5)
        elapsed = time.time() - start
        
        scores = [r[1] for r in search_results]
        results.append({
            "question_id": q['question_id'],
            "category": q['category'],
            "question": q['question'],
            "granularity": q.get('granularity', 'unknown'),
            "time": elapsed,
            "num_results": len(search_results),
            "top_score": float(max(scores)) if scores else 0.0,
            "avg_score": float(sum(scores)/len(scores)) if scores else 0.0,
            "sources": [r[2]['source'] for r in search_results]
        })
        
        if i % 20 == 0:
            logger.info(f"  Progress: {i}/{len(questions)}")
    
    return results

def main():
    questions = load_questions()
    logger.info(f"Loaded {len(questions)} questions")
    
    naive_results = run_naive_rag(questions)
    raptor_results = run_raptor(questions)
    
    output = {
        "naive": naive_results,
        "raptor": raptor_results
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
'''
    
    output_path = Path("raptor_mvp/benchmark_200q.py")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(raptor_bench_content)
    
    print(f"✓ Created: {output_path}")

def create_colbert_benchmark_script():
    """ColBERT用の200問ベンチマークスクリプトを作成"""
    colbert_bench_content = '''"""
ColBERT MVP - 200問ベンチマーク
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from naive_rag import NaiveRAG
from colbert_rag import ColBERTRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUESTIONS_FILE = Path("output/benchmark_questions_200.json")
OUTPUT_FILE = Path("output/colbert_benchmark_results_200q.json")

def load_questions():
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_naive_rag(questions):
    """Naive RAGで200問評価"""
    logger.info("Running Naive RAG on 200 questions...")
    rag = NaiveRAG(data_dir="../data/kasensabo_knowledge_base")
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
    rag = ColBERTRAG(data_dir="../data/kasensabo_knowledge_base", sample_ratio=0.5)
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
    print("\\n" + "="*80)
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
'''
    
    output_path = Path("colbert_mvp/benchmark_200q.py")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(colbert_bench_content)
    
    print(f"✓ Created: {output_path}")

def main():
    print("\n200問ベンチマーク準備スクリプト\n")
    
    # 質問ファイル準備
    prepare_question_files()
    
    # ベンチマークスクリプト作成
    print("="*80)
    print("Creating benchmark scripts")
    print("="*80)
    create_raptor_benchmark_script()
    create_colbert_benchmark_script()
    
    # 次のステップ表示
    check_benchmark_scripts()

if __name__ == "__main__":
    main()
