"""
search()のロジックをデバッグ
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from raptor_rag import RAPTORRAG
import logging

logging.basicConfig(level=logging.DEBUG)

raptor = RAPTORRAG()
raptor.load(os.path.join("output", "raptor_rag.pkl"))

query = "河川堤防の構造に関する重要な点は何か"
print(f"\nQuery: {query}\n")

results = raptor.search(query, top_k=5)

print(f"\n最終結果 ({len(results)}件):")
for i, (doc, score, meta) in enumerate(results, 1):
    print(f"{i}. Level={meta.get('level')}, Type={meta.get('type')}, Score={score:.4f}")
    print(f"   Content: {doc[:100]}...")
