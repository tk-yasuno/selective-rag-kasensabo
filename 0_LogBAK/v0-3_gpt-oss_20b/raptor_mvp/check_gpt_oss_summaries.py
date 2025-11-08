"""
gpt-oss:20bで生成された要約の品質を確認
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from raptor_rag import RAPTORRAG
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# RAPTOR読み込み
raptor = RAPTORRAG()
print("Loading RAPTOR from cache...")
raptor.load(os.path.join("output", "raptor_rag.pkl"))

print(f"\n{'='*80}")
print("gpt-oss:20bで生成された要約ノード")
print(f"{'='*80}\n")

summary_nodes = [node for node in raptor.nodes.values() if node.level > 0]
print(f"Total summary nodes: {len(summary_nodes)}\n")

for i, node in enumerate(summary_nodes, 1):
    print(f"[{i}] Level {node.level} Summary (ID: {node.node_id}):")
    print(f"Length: {len(node.content)} chars")
    print(f"Content preview:")
    print(node.content[:400])
    print(f"\n{'-'*80}\n")
