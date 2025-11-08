"""
Main Runner for Kasensabo RAPTOR MVP
河川砂防RAPTOR MVPメイン実行スクリプト
"""
import argparse
import logging
from pathlib import Path

from config import *
from naive_rag import NaiveRAG
from raptor_rag import RAPTORRAG
from benchmark import RAGBenchmark
from visualize import RAPTORVisualizer, BenchmarkVisualizer

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def build_naive_rag():
    """Naive RAGを構築"""
    logger.info("=" * 80)
    logger.info("Building Naive RAG")
    logger.info("=" * 80)
    
    rag = NaiveRAG()
    rag.load_documents(DATA_DIR)
    rag.build_index()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rag.save(OUTPUT_DIR / "naive_rag.pkl")
    
    logger.info("\n✅ Naive RAG built successfully!")


def build_raptor_rag():
    """RAPTOR RAGを構築"""
    logger.info("=" * 80)
    logger.info("Building RAPTOR RAG")
    logger.info("=" * 80)
    
    raptor = RAPTORRAG()
    raptor.load_documents(DATA_DIR)
    raptor.build_tree()
    raptor.build_index()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raptor.save(OUTPUT_DIR / "raptor_rag.pkl")
    
    logger.info("\n✅ RAPTOR RAG built successfully!")


def run_benchmark(sample_size=None):
    """ベンチマークを実行"""
    logger.info("=" * 80)
    logger.info("Running Benchmark")
    logger.info("=" * 80)
    
    benchmark = RAGBenchmark()
    benchmark.load_questions(BENCHMARK_QUESTIONS_FILE)
    benchmark.setup_rags()
    benchmark.run_benchmark(sample_size=sample_size)
    benchmark.print_summary()
    benchmark.save_results(BENCHMARK_RESULTS_FILE)
    
    logger.info("\n✅ Benchmark completed!")


def visualize():
    """可視化を実行"""
    logger.info("=" * 80)
    logger.info("Generating Visualizations")
    logger.info("=" * 80)
    
    # RAPTOR可視化
    raptor = RAPTORRAG()
    raptor.load(OUTPUT_DIR / "raptor_rag.pkl")
    
    viz = RAPTORVisualizer(raptor)
    viz.plot_tree(OUTPUT_DIR / "raptor_tree.png", max_nodes=50)
    viz.plot_level_distribution(OUTPUT_DIR / "raptor_levels.png")
    
    # ベンチマーク可視化
    if BENCHMARK_RESULTS_FILE.exists():
        bench_viz = BenchmarkVisualizer(BENCHMARK_RESULTS_FILE)
        bench_viz.plot_comparison(OUTPUT_DIR / "benchmark_comparison.png")
        bench_viz.plot_category_performance(OUTPUT_DIR / "benchmark_categories.png")
    
    logger.info("\n✅ Visualization completed!")
    logger.info(f"Output directory: {OUTPUT_DIR}")


def test_query(query: str):
    """クエリテスト"""
    logger.info("=" * 80)
    logger.info(f"Testing Query: {query}")
    logger.info("=" * 80)
    
    # Naive RAG
    logger.info("\n--- Naive RAG ---")
    naive = NaiveRAG()
    naive.load(OUTPUT_DIR / "naive_rag.pkl")
    naive_results = naive.search(query, top_k=3)
    
    for i, (chunk, score, meta) in enumerate(naive_results, 1):
        print(f"{i}. Score: {score:.4f} | Source: {meta['source']}")
        print(f"   {chunk[:150]}...\n")
    
    # RAPTOR RAG
    logger.info("\n--- RAPTOR RAG ---")
    raptor = RAPTORRAG()
    raptor.load(OUTPUT_DIR / "raptor_rag.pkl")
    raptor_results = raptor.search(query, top_k=3)
    
    for i, (chunk, score, meta) in enumerate(raptor_results, 1):
        print(f"{i}. Score: {score:.4f} | Source: {meta.get('source', 'N/A')} | Level: {meta.get('level', 'N/A')}")
        print(f"   {chunk[:150]}...\n")


def main():
    parser = argparse.ArgumentParser(description="Kasensabo RAPTOR MVP")
    parser.add_argument('command', choices=['build', 'benchmark', 'visualize', 'test', 'all'],
                       help='Command to execute')
    parser.add_argument('--query', type=str, help='Query for test command')
    parser.add_argument('--sample', type=int, help='Sample size for benchmark')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        build_naive_rag()
        build_raptor_rag()
    
    elif args.command == 'benchmark':
        run_benchmark(sample_size=args.sample)
    
    elif args.command == 'visualize':
        visualize()
    
    elif args.command == 'test':
        if not args.query:
            args.query = "堤防と護岸の機能的違いは？"
        test_query(args.query)
    
    elif args.command == 'all':
        build_naive_rag()
        build_raptor_rag()
        run_benchmark(sample_size=args.sample or 10)
        visualize()


if __name__ == "__main__":
    main()
