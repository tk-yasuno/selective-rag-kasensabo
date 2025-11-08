"""要約ノードの検索スコアをデバッグ"""
from raptor_rag import RAPTORRAG
import numpy as np

# RAPTORをロード
r = RAPTORRAG()
r.load('output/raptor_rag.pkl')

# テストクエリ
query = '河川堤防の設計基準は何ですか'
print(f"Query: {query}\n")

# クエリ埋め込み
query_embedding = r.embedding_model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
)

# 全ノード検索（大きめに）
search_k = 100
scores, indices = r.faiss_index.search(query_embedding.astype('float32'), search_k)

# ノードIDに変換
index_to_node_id = {idx: nid for nid, idx in r.node_id_to_index.items()}

print(f"Top {search_k} results:\n")
print(f"{'Rank':<6}{'Level':<8}{'Score':<10}{'NodeID':<20}{'Content':<60}")
print("="*104)

for i, (score, idx) in enumerate(zip(scores[0][:100], indices[0][:100])):
    if idx in index_to_node_id:
        node_id = index_to_node_id[idx]
        node = r.nodes[node_id]
        content_preview = node.content[:57] + "..." if len(node.content) > 60 else node.content
        print(f"{i+1:<6}{node.level:<8}{score:<10.4f}{node_id:<20}{content_preview}")
