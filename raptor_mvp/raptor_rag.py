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
from sklearn.metrics import silhouette_score, davies_bouldin_score
from dataclasses import dataclass
import time
import logging
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama

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
        
        # LLM要約の初期化
        self.llm_model = None
        self.llm_tokenizer = None
        if USE_LLM_SUMMARY:
            self._init_llm()
    
    def _init_llm(self):
        """LLM要約モデルの初期化"""
        if USE_OLLAMA:
            try:
                logger.info(f"Using Ollama model: {OLLAMA_MODEL}")
                # Ollamaが起動しているか確認
                ollama.list()
                logger.info(f"✓ Ollama connection successful")
                self.use_ollama = True
            except Exception as e:
                logger.warning(f"Failed to connect to Ollama: {e}. Falling back to simple concatenation.")
                self.llm_model = None
                self.llm_tokenizer = None
                self.use_ollama = False
        else:
            try:
                logger.info(f"Loading LLM model: {LLM_MODEL}")
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    LLM_MODEL,
                    trust_remote_code=True
                )
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                self.llm_model.eval()
                self.use_ollama = False
                logger.info(f"✓ LLM model loaded successfully on {DEVICE}")
            except Exception as e:
                logger.warning(f"Failed to load LLM: {e}. Falling back to simple concatenation.")
                self.llm_model = None
                self.llm_tokenizer = None
                self.use_ollama = False
        
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
                # 上位レベルでは最小サイズ制約を緩和
                min_size = 2 if len(current_level_nodes) <= 10 else MIN_CLUSTER_SIZE
                if len(cluster_node_ids) < min_size:
                    continue
                
                # 要約生成
                summary = self._create_summary(cluster_node_ids)
                
                # 要約の埋め込み: 要約テキストから生成
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
        """ノードをクラスタリング（FAISS k-means使用）"""
        embeddings = np.array([self.nodes[nid].embedding for nid in node_ids]).astype('float32')
        
        # クラスタ数を評価
        min_k = MIN_CLUSTERS
        max_k = min(MAX_CLUSTERS, len(node_ids))
        
        if len(node_ids) < min_k:
            return [node_ids]
        
        # 小規模データの場合は最大クラスタ数を制限
        if len(node_ids) < 30:
            max_k = min(max_k, len(node_ids) // 2)
        
        best_k = min_k
        best_score = -np.inf
        
        logger.info(f"  Evaluating k={min_k}~{max_k}...")
        
        for k in range(min_k, max_k + 1):
            if k >= len(node_ids):
                break
            
            # FAISS k-meansでクラスタリング
            d = embeddings.shape[1]
            kmeans = faiss.Kmeans(d, k, niter=20, verbose=False, gpu=False)
            kmeans.train(embeddings)
            _, labels = kmeans.index.search(embeddings, 1)
            labels = labels.flatten()
            
            # Silhouette Score (高いほど良い)
            sil_score = silhouette_score(embeddings, labels)
            
            # Davies-Bouldin Index (低いほど良い)
            dbi_score = davies_bouldin_score(embeddings, labels)
            
            # 正規化して組み合わせ
            sil_normalized = sil_score
            dbi_normalized = 1.0 / (1.0 + dbi_score)
            
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
        d = embeddings.shape[1]
        kmeans = faiss.Kmeans(d, best_k, niter=20, verbose=False, gpu=False)
        kmeans.train(embeddings)
        _, labels = kmeans.index.search(embeddings, 1)
        labels = labels.flatten()
        
        # クラスタごとにノードIDをグループ化
        clusters = [[] for _ in range(best_k)]
        for node_id, label in zip(node_ids, labels):
            clusters[label].append(node_id)
        
        return clusters
    
    def _create_summary(self, node_ids: List[str]) -> str:
        """クラスタから要約を生成（LLM使用）"""
        # 子ノードのコンテンツを収集（橋梁診断ロジック完全準拠）
        texts = [self.nodes[nid].content for nid in node_ids]  # 全件使用
        combined_text = "\n\n".join(texts)
        
        # 橋梁診断と同じ入力長制限
        max_input_length = 8000  # 橋梁診断の成功設定
        
        if len(combined_text) > max_input_length:
            # 均等サンプリング（橋梁診断ロジック）
            sample_ratio = max_input_length / len(combined_text)
            sampled_texts = []
            for text in texts:
                sample_len = int(len(text) * sample_ratio)
                if sample_len > 0:
                    sampled_texts.append(text[:sample_len])
            combined_text = "\n\n".join(sampled_texts)[:max_input_length]
        
        # Ollama使用
        if USE_OLLAMA and hasattr(self, 'use_ollama') and self.use_ollama:
            try:
                # 橋梁診断で成功したプロンプト形式（河川砂防用に調整）
                prompt = f"""以下は河川砂防ダム技術基準に関する複数のドキュメントです。

【要約タスク】
- 施設の構造・管理基準、点検項目、設計手法を要約してください
- 400-500文字で簡潔にまとめてください
- 箇条書きではなく、段落形式で記述してください

【テキスト】
{combined_text}

【要約】"""
                
                response = ollama.generate(
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    options={
                        'temperature': 0,  # 橋梁診断と同じ決定的設定
                        'top_p': 0.9,
                        'num_predict': 500,  # 橋梁診断と同じ
                        'num_ctx': 16384  # 橋梁診断と同じ大容量
                    },
                    keep_alive='10m'
                )
                
                summary = response['response'].strip()
                
                if summary and len(summary) > 100:  # 橋梁診断基準
                    return summary
            
            except Exception as e:
                logger.warning(f"Ollama summarization failed: {e}, falling back to concatenation")
        
        # HuggingFace transformers使用
        elif self.llm_model is not None and self.llm_tokenizer is not None:
            try:
                prompt = f"""以下の河川砂防ダム技術基準の文書を、専門的かつ簡潔に要約してください。
重要な技術用語、基準値、設計手法を保持してください。

文書:
{combined_text[:1500]}

要約:"""
                
                inputs = self.llm_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.llm_model.device)
                
                with torch.no_grad():
                    outputs = self.llm_model.generate(
                        **inputs,
                        max_new_tokens=LLM_MAX_NEW_TOKENS,
                        temperature=LLM_TEMPERATURE,
                        top_p=LLM_TOP_P,
                        repetition_penalty=LLM_REPETITION_PENALTY,
                        do_sample=True,
                        pad_token_id=self.llm_tokenizer.pad_token_id,
                        eos_token_id=self.llm_tokenizer.eos_token_id
                    )
                
                summary = self.llm_tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                if summary and len(summary) > 10:
                    return summary
            
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}, falling back to concatenation")
        
        # フォールバック：簡易結合
        if len(combined_text) > 800:
            combined_text = combined_text[:800] + "..."
        
        return combined_text
    
    def build_index(self) -> None:
        """全ノードからFAISSインデックスを構築"""
        logger.info("Building FAISS index from all nodes...")
        
        # ノードIDリストを取得（順序を保持）
        node_ids = list(self.nodes.keys())
        embeddings = np.array([self.nodes[nid].embedding for nid in node_ids])
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
        
        # node_id_to_indexを更新（重要！）
        self.node_id_to_index = {nid: idx for idx, nid in enumerate(node_ids)}
        
        # 要約ノードの数を確認
        summary_count = sum(1 for nid in node_ids if not self.nodes[nid].is_leaf)
        logger.info(f"Index built: {len(node_ids)} nodes (including {summary_count} summary nodes), dim={dimension}")
        
        logger.info(f"Index built: {len(node_ids)} nodes, dim={dimension}")
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[str, float, dict]]:
        """階層的検索（Tree-traversal戦略）"""
        if self.faiss_index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # クエリ埋め込み
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # 全ノードから検索（要約ノードは数が少なく低スコアのため全件検索）
        search_k = len(self.nodes)  # 全ノードから検索して要約を確実に取得
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), search_k)
        
        # ノードIDに変換
        index_to_node_id = {idx: nid for nid, idx in self.node_id_to_index.items()}
        
        # 階層ごとにスコアを集計
        level_results = {}  # level -> [(node_id, score, node)]
        
        for score, idx in zip(scores[0], indices[0]):
            if idx in index_to_node_id:
                node_id = index_to_node_id[idx]
                node = self.nodes[node_id]
                
                if node.level not in level_results:
                    level_results[node.level] = []
                
                level_results[node.level].append((node_id, float(score), node))
        
        # デバッグ: 各レベルの検索結果数を確認
        logger.debug(f"Level distribution in search results: {dict((k, len(v)) for k, v in level_results.items())}")
        
        # 階層統合戦略: 固定比率ミックス（RAPTOR論文準拠+橋梁診断成功ロジック）
        # 戦略: 要約は必ず1つ含め、スコアソートせず構造的配置
        # 理由: 要約スコアはリーフの半分程度だが、文脈理解に不可欠
        
        results = []
        
        # ステップ1: 最高レベルの要約を1つ確保（Level 2 > Level 1の優先順位）
        best_summary = None
        
        for level in [2, 1]:
            if level in level_results and len(level_results[level]) > 0:
                node_id, score, node = max(level_results[level], key=lambda x: x[1])
                best_summary = (
                    node.content,
                    score,  # スコアブーストしない（構造的に強制配置）
                    {**node.metadata, 'level': level, 'node_id': node_id, 'type': f'summary_l{level}'}
                )
                break  # 最初に見つかった最高レベルで確定
        
        if best_summary:
            results.append(best_summary)
            logger.info(f"Added summary from Level {best_summary[2]['level']}")
        
        # ステップ2: 残りをリーフノードの上位でフ埋める
        if 0 in level_results:
            leaf_candidates = sorted(level_results[0], key=lambda x: x[1], reverse=True)
            remaining = top_k - len(results)
            
            for node_id, score, node in leaf_candidates[:remaining]:
                results.append((
                    node.content,
                    score,
                    {**node.metadata, 'level': 0, 'node_id': node_id, 'type': 'leaf'}
                ))
            
            logger.info(f"Added {len(results)-1} leaf nodes (top scores: {leaf_candidates[0][1]:.4f} - {leaf_candidates[remaining-1][1]:.4f})")
        
        # スコアソートせず、構造順（要約→リーフ）で返す
        return results[:top_k]
    
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
    
    def load(self, output_path) -> None:
        """保存したツリーを読み込み"""
        output_path = Path(output_path)
        
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
