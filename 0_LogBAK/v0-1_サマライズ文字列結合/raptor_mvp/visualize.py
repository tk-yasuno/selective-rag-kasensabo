"""
Visualization for RAPTOR Tree and Benchmark Results
RAPTOR可視化とベンチマーク結果の表示
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

import networkx as nx
from pathlib import Path
import numpy as np
from typing import Dict
import pickle

from config import *
from raptor_rag import RAPTORRAG, RAPTORNode

import logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class RAPTORVisualizer:
    """RAPTOR可視化"""
    
    def __init__(self, raptor: RAPTORRAG):
        self.raptor = raptor
        self.graph = nx.DiGraph()
    
    def build_graph(self) -> None:
        """ツリー構造からグラフを構築"""
        logger.info("Building graph from RAPTOR tree...")
        
        for node_id, node in self.raptor.nodes.items():
            # ノード追加
            label = f"L{node.level}"
            if node.is_leaf:
                label += f"\n{node.metadata.get('source', '')[:20]}"
            else:
                label += f"\nC{node.cluster_id}"
            
            self.graph.add_node(
                node_id,
                level=node.level,
                is_leaf=node.is_leaf,
                label=label
            )
            
            # エッジ追加
            if node.parent_id:
                self.graph.add_edge(node.parent_id, node_id)
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def plot_tree(self, output_file: Path, max_nodes: int = 100) -> None:
        """ツリーを可視化"""
        if self.graph.number_of_nodes() == 0:
            self.build_graph()
        
        # ノード数が多い場合はサンプリング
        if self.graph.number_of_nodes() > max_nodes:
            logger.warning(f"Too many nodes ({self.graph.number_of_nodes()}). Sampling {max_nodes} nodes...")
            # 非リーフノードを優先
            non_leaf_nodes = [n for n, d in self.graph.nodes(data=True) if not d.get('is_leaf', True)]
            if len(non_leaf_nodes) > max_nodes:
                sampled_nodes = non_leaf_nodes[:max_nodes]
            else:
                leaf_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('is_leaf', False)]
                sampled_nodes = non_leaf_nodes + leaf_nodes[:max_nodes - len(non_leaf_nodes)]
            
            subgraph = self.graph.subgraph(sampled_nodes)
        else:
            subgraph = self.graph
        
        # レイアウト計算
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
        
        # プロット
        plt.figure(figsize=(20, 15))
        
        # ノードの色分け（レベルごと）
        levels = [d.get('level', 0) for n, d in subgraph.nodes(data=True)]
        
        nx.draw_networkx_nodes(
            subgraph, pos,
            node_color=levels,
            node_size=200,
            cmap=plt.cm.viridis,
            alpha=0.8
        )
        
        nx.draw_networkx_edges(
            subgraph, pos,
            edge_color='gray',
            alpha=0.3,
            arrows=True,
            arrowsize=10
        )
        
        # ラベル（小さめに表示）
        labels = nx.get_node_attributes(subgraph, 'label')
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=6)
        
        plt.title('RAPTOR Tree Structure', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Tree visualization saved to {output_file}")
        plt.close()
    
    def plot_level_distribution(self, output_file: Path) -> None:
        """レベルごとのノード数分布を可視化"""
        levels = {}
        for node in self.raptor.nodes.values():
            if node.level not in levels:
                levels[node.level] = 0
            levels[node.level] += 1
        
        plt.figure(figsize=(10, 6))
        plt.bar(levels.keys(), levels.values(), color='skyblue', edgecolor='black')
        plt.xlabel('Tree Level', fontsize=12)
        plt.ylabel('Number of Nodes', fontsize=12)
        plt.title('RAPTOR Tree - Node Distribution by Level', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Level distribution saved to {output_file}")
        plt.close()


class BenchmarkVisualizer:
    """ベンチマーク結果の可視化"""
    
    def __init__(self, results_file: Path):
        with open(results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
    
    def plot_comparison(self, output_file: Path) -> None:
        """Naive vs RAPTOR比較チャート"""
        comp = self.results['comparison']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 時間比較
        axes[0].bar(['Naive', 'RAPTOR'], 
                    [comp['naive']['avg_time']*1000, comp['raptor']['avg_time']*1000],
                    color=['coral', 'lightgreen'])
        axes[0].set_ylabel('Average Time (ms)')
        axes[0].set_title('Response Time')
        axes[0].grid(axis='y', alpha=0.3)
        
        # スコア比較
        axes[1].bar(['Naive', 'RAPTOR'],
                    [comp['naive']['avg_score'], comp['raptor']['avg_score']],
                    color=['coral', 'lightgreen'])
        axes[1].set_ylabel('Average Similarity Score')
        axes[1].set_title('Retrieval Quality')
        axes[1].grid(axis='y', alpha=0.3)
        
        # トップスコア比較
        axes[2].bar(['Naive', 'RAPTOR'],
                    [comp['naive']['top_score'], comp['raptor']['top_score']],
                    color=['coral', 'lightgreen'])
        axes[2].set_ylabel('Top-1 Similarity Score')
        axes[2].set_title('Best Match Quality')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Naive RAG vs RAPTOR Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Comparison chart saved to {output_file}")
        plt.close()
    
    def plot_category_performance(self, output_file: Path) -> None:
        """カテゴリ別性能比較"""
        naive_by_category = {}
        raptor_by_category = {}
        
        for result in self.results['naive']:
            cat = result['category']
            if cat not in naive_by_category:
                naive_by_category[cat] = []
            naive_by_category[cat].append(result['avg_score'])
        
        for result in self.results['raptor']:
            cat = result['category']
            if cat not in raptor_by_category:
                raptor_by_category[cat] = []
            raptor_by_category[cat].append(result['avg_score'])
        
        categories = list(naive_by_category.keys())
        naive_scores = [np.mean(naive_by_category[cat]) for cat in categories]
        raptor_scores = [np.mean(raptor_by_category[cat]) for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.figure(figsize=(14, 6))
        plt.bar(x - width/2, naive_scores, width, label='Naive RAG', color='coral')
        plt.bar(x + width/2, raptor_scores, width, label='RAPTOR', color='lightgreen')
        
        plt.xlabel('Category')
        plt.ylabel('Average Similarity Score')
        plt.title('Performance by Question Category')
        plt.xticks(x, [cat[:15] + '...' if len(cat) > 15 else cat for cat in categories], rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Category performance chart saved to {output_file}")
        plt.close()


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # RAPTOR可視化
    logger.info("=== RAPTOR Tree Visualization ===")
    raptor = RAPTORRAG()
    raptor.load(OUTPUT_DIR / "raptor_rag.pkl")
    
    viz = RAPTORVisualizer(raptor)
    viz.plot_tree(OUTPUT_DIR / "raptor_tree.png", max_nodes=50)
    viz.plot_level_distribution(OUTPUT_DIR / "raptor_levels.png")
    
    # ベンチマーク可視化
    if BENCHMARK_RESULTS_FILE.exists():
        logger.info("\n=== Benchmark Results Visualization ===")
        bench_viz = BenchmarkVisualizer(BENCHMARK_RESULTS_FILE)
        bench_viz.plot_comparison(OUTPUT_DIR / "benchmark_comparison.png")
        bench_viz.plot_category_performance(OUTPUT_DIR / "benchmark_categories.png")
    
    logger.info("\nVisualization complete!")
