"""RAPTORの要約ノードを検証"""
import pickle
from pathlib import Path

# RAPTORをロード
with open('output/raptor_rag.pkl', 'rb') as f:
    data = pickle.load(f)

nodes = data['nodes']

# 要約ノードを抽出
summary_nodes = [n for n in nodes.values() if not n.is_leaf]

print(f"Total nodes: {len(nodes)}")
print(f"Summary nodes: {len(summary_nodes)}")
print(f"Leaf nodes: {len(nodes) - len(summary_nodes)}")
print("\n" + "="*80)

# 各レベルの要約を表示
for level in [1, 2]:
    level_nodes = [n for n in summary_nodes if n.level == level]
    print(f"\n=== Level {level} Summaries ({len(level_nodes)} nodes) ===\n")
    
    for i, node in enumerate(level_nodes[:2]):  # 最初の2つを表示
        print(f"Node {i+1} (ID: {node.node_id}):")
        print(f"Content length: {len(node.content)} chars")
        print(f"First 400 chars:\n{node.content[:400]}...")
        print("\n" + "-"*80 + "\n")
