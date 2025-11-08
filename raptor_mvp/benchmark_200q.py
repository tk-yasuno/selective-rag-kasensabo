"""
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
    rag = NaiveRAG()
    rag.load_documents(Path("../data/kasensabo_knowledge_base"))
    rag.build_index()
    
    results = []
    for i, q in enumerate(questions, 1):
        start = time.time()
        search_results = rag.search(q['question'], top_k=5)
        elapsed = time.time() - start
        
        scores = [r[1] for r in search_results]
        # メタデータのsourceキーを取得（存在しない場合はunknown）
        sources = [r[2].get('source', 'unknown') if len(r) > 2 and isinstance(r[2], dict) else 'unknown' for r in search_results]
        
        results.append({
            "question_id": q['question_id'],
            "category": q['category'],
            "question": q['question'],
            "granularity": q.get('granularity', 'unknown'),
            "time": elapsed,
            "num_results": len(search_results),
            "top_score": float(max(scores)) if scores else 0.0,
            "avg_score": float(sum(scores)/len(scores)) if scores else 0.0,
            "sources": sources
        })
        
        if i % 20 == 0:
            logger.info(f"  Progress: {i}/{len(questions)}")
    
    return results

def run_raptor(questions):
    """RAPTORで200問評価"""
    logger.info("Running RAPTOR on 200 questions...")
    rag = RAPTORRAG()
    
    raptor_pkl = Path("output/raptor_rag.pkl")
    if raptor_pkl.exists():
        logger.info("Loading existing RAPTOR tree...")
        rag.load(raptor_pkl)
    else:
        logger.info("Building RAPTOR tree...")
        rag.load_documents(Path("../data/kasensabo_knowledge_base"))
        rag.build_tree()
        rag.save(raptor_pkl)
    
    results = []
    for i, q in enumerate(questions, 1):
        start = time.time()
        search_results = rag.search(q['question'], top_k=5)
        elapsed = time.time() - start
        
        scores = [r[1] for r in search_results]
        # メタデータのsourceキーを取得（存在しない場合はunknown）
        sources = [r[2].get('source', 'unknown') if len(r) > 2 and isinstance(r[2], dict) else 'unknown' for r in search_results]
        
        results.append({
            "question_id": q['question_id'],
            "category": q['category'],
            "question": q['question'],
            "granularity": q.get('granularity', 'unknown'),
            "time": elapsed,
            "num_results": len(search_results),
            "top_score": float(max(scores)) if scores else 0.0,
            "avg_score": float(sum(scores)/len(scores)) if scores else 0.0,
            "sources": sources
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
