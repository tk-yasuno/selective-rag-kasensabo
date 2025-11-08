"""
検索結果における階層分布を確認
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from raptor_rag import RAPTORRAG
from config import *
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# RAPTOR読み込み
raptor = RAPTORRAG()
print("Loading RAPTOR from cache...")
raptor.load(os.path.join(OUTPUT_DIR, "raptor_rag.pkl"))

# テストクエリ
queries = [
    "河川堤防の構造に関する重要な点は何か",
    "砂防ダムの点検で重視すべき項目は",
    "流量観測の手法にはどのような方法があるか",
]

top_k = 5
print(f"\n検索結果の階層分布 (top_k={top_k}):")
print("="*80)

for query in queries:
    print(f"\nQuery: {query}")
    results = raptor.search(query, top_k=top_k)
    
    # レベル別カウント
    level_count = {}
    for doc, score, meta in results:
        level = meta.get('level', 0)
        level_count[level] = level_count.get(level, 0) + 1
    
    print(f"  Level 分布: {level_count}")
    print(f"  Top 5 結果:")
    for i, (doc, score, meta) in enumerate(results[:5], 1):
        print(f"    {i}. Level={meta.get('level', 0)}, Type={meta.get('type', 'unknown')}, Score={score:.4f}")

print("\n" + "="*80)
print("要約ノード統計:")
summary_usage = {}
for query in queries:
    results = raptor.search(query, top_k=top_k)
    for doc, score, meta in results:
        if 'summary' in meta.get('type', ''):
            summary_usage[query] = summary_usage.get(query, 0) + 1

print(f"要約ノードが含まれたクエリ数: {len(summary_usage)}/{len(queries)}")
if summary_usage:
    print(f"平均要約ノード数/クエリ: {sum(summary_usage.values()) / len(queries):.1f}")
