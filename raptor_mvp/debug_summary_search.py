"""要約ノードが検索結果に含まれない理由をデバッグ"""
from raptor_rag import RAPTORRAG
import numpy as np

# RAPTORをロード
r = RAPTORRAG()
r.load('output/raptor_rag.pkl')

# 要約ノードを取得
summary_nodes = [n for n_id, n in r.nodes.items() if not n.is_leaf]
print(f"Summary nodes found: {len(summary_nodes)}")

# テストクエリ
query = '河川堤防の設計基準は何ですか'
print(f"\nQuery: {query}")

# クエリ埋め込み
query_embedding = r.embedding_model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("\n=== Summary Node Scores ===")
for node in summary_nodes:
    # ノードの埋め込みとクエリの類似度を直接計算
    similarity = np.dot(query_embedding[0], node.embedding)
    print(f"Level {node.level}, ID: {node.node_id}")
    print(f"  Score: {similarity:.4f}")
    print(f"  Content preview: {node.content[:100]}...")
    print()

# 検索実行
print("\n=== Actual Search Results ===")
results = r.search(query, top_k=10)

for i, (content, score, metadata) in enumerate(results):
    print(f"{i+1}. Level={metadata['level']}, Type={metadata['type']}, Score={score:.4f}")
    print(f"   NodeID: {metadata['node_id']}")
    print()
