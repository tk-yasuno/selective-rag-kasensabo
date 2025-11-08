"""
RAPTOR RAG Implementation
階層的クラスタリングと再ランキングを持つRAG
"""
import numpy as np
import faiss
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from dataclasses import dataclass
import time
import logging
import pickle

from config import *

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


@dataclass
class RAPTORNode:
    """RAPTOR Tree Node"""
    node_id: str
    parent_id: Optional[str]
    children: List[str]
    level: int
    content: str
    summary: str
    is_leaf: bool
    cluster_id: Optional[int]
    embedding: np.ndarray
    metadata: dict


class RAPTORRAG:
    """階層的RAPTOR RAG"""
    
    def __init__(self):
        logger.info(f"Initializing RAPTOR RAG with {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        self.nodes: Dict[str, RAPTORNode] = {}
        self.faiss_index = None
        self.node_id_to_index = {}
        
    def load_documents(self, data_dir: Path) -> None:
        """文書を読み込みチャンク化してリーフノードを作成"""
        logger.info(f"Loading documents from {data_dir}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "、", " ", ""]
        )
        
        all_chunks = []
        all_metadata = []
        
        for file_path in data_dir.glob("*.md"):
            logger.info(f"  Processing {file_path.name}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = text_splitter.split_text(content)
            
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'source': file_path.name,
                    'chunk_id': idx
                })
        
        logger.info(f"Loaded {len(all_chunks)} chunks")
        
        # リーフノード作成
        logger.info("Creating leaf nodes...")
        embeddings = self.embedding_model.encode(
            all_chunks,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        for idx, (chunk, embedding, metadata) in enumerate(zip(all_chunks, embeddings, all_metadata)):
            node_id = f"leaf_{idx:05d}"
            node = RAPTORNode(
                node_id=node_id,
                parent_id=None,
                children=[],
                level=0,
                content=chunk,
                summary=chunk[:200],
                is_leaf=True,
                cluster_id=None,
                embedding=embedding,
                metadata=metadata
            )
            self.nodes[node_id] = node
        
        logger.info(f"Created {len(self.nodes)} leaf nodes")
    
    def build_tree(self) -> None:
        """階層的ツリーを構築"""
        logger.info(f"Building RAPTOR tree (max_depth={MAX_TREE_DEPTH})")
        
        current_level_nodes = [nid for nid in self.nodes.keys() if self.nodes[nid].level == 0]
        current_depth = 1
        
        while current_depth <= MAX_TREE_DEPTH and len(current_level_nodes) > MIN_CLUSTER_SIZE:
            logger.info(f"\n=== Level {current_depth} ===")
            logger.info(f"Input nodes: {len(current_level_nodes)}")
            
            # クラスタリング
            clusters = self._cluster_nodes(current_level_nodes, current_depth)
            logger.info(f"Created {len(clusters)} clusters")
            
            # 各クラスタから要約ノードを作成
            next_level_nodes = []
            
            for cluster_idx, cluster_node_ids in enumerate(clusters):
                if len(cluster_node_ids) < MIN_CLUSTER_SIZE:
                    continue
                
                # 要約生成
                summary = self._create_summary(cluster_node_ids)
                
                # 要約の埋め込み
                summary_embedding = self.embedding_model.encode(
                    [summary],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )[0]
                
                # 要約ノード作成
                node_id = f"summary_l{current_depth}_c{cluster_idx:03d}"
                summary_node = RAPTORNode(
                    node_id=node_id,
                    parent_id=None,
                    children=cluster_node_ids,
                    level=current_depth,
                    content=summary,
                    summary=summary[:200],
                    is_leaf=False,
                    cluster_id=cluster_idx,
                    embedding=summary_embedding,
                    metadata={'cluster_size': len(cluster_node_ids)}
                )
                
                self.nodes[node_id] = summary_node
                
                # 子ノードに親を設定
                for child_id in cluster_node_ids:
                    self.nodes[child_id].parent_id = node_id
                
                next_level_nodes.append(node_id)
            
            logger.info(f"Created {len(next_level_nodes)} summary nodes")
            
            if len(next_level_nodes) == 0:
                break
            
            current_level_nodes = next_level_nodes
            current_depth += 1
        
        logger.info(f"\nTree building complete. Total nodes: {len(self.nodes)}")
        self._print_tree_stats()
    
    def _cluster_nodes(self, node_ids: List[str], depth: int) -> List[List[str]]:
        """ノードをクラスタリング"""
        embeddings = np.array([self.nodes[nid].embedding for nid in node_ids])
        
        # クラスタ数を評価
        min_k = MIN_CLUSTERS
        max_k = min(MAX_CLUSTERS, len(node_ids))
        
        if len(node_ids) < min_k:
            return [node_ids]
        
        best_k = min_k
        best_score = -np.inf
        
        logger.info(f"  Evaluating k={min_k}~{max_k}...")
        
        for k in range(min_k, max_k + 1):
            if k >= len(node_ids):
                break
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            # Silhouette Score (高いほど良い)
            sil_score = silhouette_score(embeddings, labels)
            
            # Davies-Bouldin Index (低いほど良い)
            dbi_score = davies_bouldin_score(embeddings, labels)
            
            # 正規化して組み合わせ
            # Silhouette: 0~1 → そのまま使用
            # DBI: 0~∞ → 逆数を取って正規化
            sil_normalized = sil_score
            dbi_normalized = 1.0 / (1.0 + dbi_score)  # 0~1に正規化
            
            combined_score = (
                METRIC_WEIGHTS['silhouette'] * sil_normalized +
                METRIC_WEIGHTS['dbi'] * dbi_normalized
            )
            
            logger.info(f"    k={k}: Sil={sil_score:.4f}, DBI={dbi_score:.4f}, Combined={combined_score:.4f}")
            
            if combined_score > best_score:
                best_score = combined_score
                best_k = k
        
        logger.info(f"  → Selected k={best_k}")
        
        # 最適なkでクラスタリング
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # クラスタごとにノードIDをグループ化
        clusters = [[] for _ in range(best_k)]
        for node_id, label in zip(node_ids, labels):
            clusters[label].append(node_id)
        
        return clusters
    
    def _create_summary(self, node_ids: List[str]) -> str:
        """クラスタから要約を生成（簡易版：先頭部分を結合）"""
        texts = [self.nodes[nid].content for nid in node_ids[:5]]  # 最大5件
        combined = "\n\n".join([t[:200] for t in texts])
        
        if len(combined) > 800:
            combined = combined[:800] + "..."
        
        return combined
    
    def build_index(self) -> None:
        """全ノードからFAISSインデックスを構築"""
        logger.info("Building FAISS index from all nodes...")
        
        node_ids = list(self.nodes.keys())
        embeddings = np.array([self.nodes[nid].embedding for nid in node_ids])
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
        
        self.node_id_to_index = {nid: idx for idx, nid in enumerate(node_ids)}
        
        logger.info(f"Index built: {len(node_ids)} nodes, dim={dimension}")
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[str, float, dict]]:
        """階層的検索"""
        if self.faiss_index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # クエリ埋め込み
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # 検索（階層を考慮して多めに取得）
        search_k = top_k * 3
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), search_k)
        
        # ノードIDに変換
        index_to_node_id = {idx: nid for nid, idx in self.node_id_to_index.items()}
        
        results = []
        leaf_count = 0
        
        for score, idx in zip(scores[0], indices[0]):
            if idx in index_to_node_id:
                node_id = index_to_node_id[idx]
                node = self.nodes[node_id]
                
                # リーフノードを優先
                if node.is_leaf:
                    results.append((
                        node.content,
                        float(score),
                        {**node.metadata, 'level': node.level, 'node_id': node_id}
                    ))
                    leaf_count += 1
                    
                    if leaf_count >= top_k:
                        break
        
        return results
    
    def _print_tree_stats(self) -> None:
        """ツリー統計を表示"""
        levels = {}
        for node in self.nodes.values():
            if node.level not in levels:
                levels[node.level] = 0
            levels[node.level] += 1
        
        logger.info("\nTree Statistics:")
        for level in sorted(levels.keys()):
            logger.info(f"  Level {level}: {levels[level]} nodes")
    
    def save(self, output_path: Path) -> None:
        """ツリーとインデックスを保存"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump({
                'nodes': self.nodes,
                'node_id_to_index': self.node_id_to_index,
                'model_name': EMBEDDING_MODEL
            }, f)
        
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(output_path.with_suffix('.faiss')))
        
        logger.info(f"Saved to {output_path}")
    
    def load(self, output_path: Path) -> None:
        """保存したツリーを読み込み"""
        with open(output_path, 'rb') as f:
            data = pickle.load(f)
        
        self.nodes = data['nodes']
        self.node_id_to_index = data['node_id_to_index']
        
        faiss_path = output_path.with_suffix('.faiss')
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
        
        logger.info(f"Loaded from {output_path}")


if __name__ == "__main__":
    # テスト実行
    raptor = RAPTORRAG()
    raptor.load_documents(DATA_DIR)
    raptor.build_tree()
    raptor.build_index()
    
    # サンプルクエリ
    query = "堤防と護岸の機能的違いは？"
    results = raptor.search(query, top_k=3)
    
    print(f"\nQuery: {query}\n")
    for i, (chunk, score, meta) in enumerate(results, 1):
        print(f"{i}. Score: {score:.4f} | Source: {meta.get('source', 'N/A')} | Level: {meta.get('level', 'N/A')}")
        print(f"   {chunk[:150]}...\n")
    
    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raptor.save(OUTPUT_DIR / "raptor_rag.pkl")
