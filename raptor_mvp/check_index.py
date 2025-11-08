"""要約ノードの存在確認"""
import pickle
import numpy as np

data = pickle.load(open('output/raptor_rag.pkl', 'rb'))
nodes = data['nodes']

summary_nodes = [n for n in nodes.values() if not n.is_leaf]
leaf_nodes = [n for n in nodes.values() if n.is_leaf]

print(f"Total nodes: {len(nodes)}")
print(f"Summary nodes: {len(summary_nodes)}")
print(f"Leaf nodes: {len(leaf_nodes)}")

if summary_nodes:
    print(f"\nSample summary node:")
    s = summary_nodes[0]
    print(f"  ID: {s.node_id}")
    print(f"  Level: {s.level}")
    print(f"  Embedding shape: {s.embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(s.embedding):.4f}")
    print(f"  Content length: {len(s.content)}")
    
# node_id_to_index確認
node_id_to_index = data['node_id_to_index']
print(f"\nnode_id_to_index entries: {len(node_id_to_index)}")

# 要約ノードがインデックスに含まれているか
summary_in_index = sum(1 for s in summary_nodes if s.node_id in node_id_to_index)
print(f"Summary nodes in index: {summary_in_index}/{len(summary_nodes)}")
