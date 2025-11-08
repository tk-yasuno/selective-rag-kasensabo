"""
3つのRAGシステム(Naive RAG, RAPTOR, ColBERT)のスコアと速度を比較
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# データファイルパス
RAPTOR_RESULTS = Path("raptor_mvp/output/benchmark_results.json")
COLBERT_RESULTS = Path("colbert_mvp/output/colbert_benchmark_results.json")
SELECTIVE_RESULTS = Path("selective_rag/output/selective_rag_benchmark_results.json")

def load_data():
    """各システムの結果データを読み込む"""
    
    # RAPTOR結果
    with open(RAPTOR_RESULTS, 'r', encoding='utf-8') as f:
        raptor_data = json.load(f)
    
    # ColBERT結果
    with open(COLBERT_RESULTS, 'r', encoding='utf-8') as f:
        colbert_data = json.load(f)
    
    # Selective RAG結果
    with open(SELECTIVE_RESULTS, 'r', encoding='utf-8') as f:
        selective_data = json.load(f)
    
    # RAPTOR統計計算
    raptor_scores = [item['avg_score'] for item in raptor_data['raptor']]
    raptor_top_scores = [item['top_score'] for item in raptor_data['raptor']]
    raptor_times = [item['time'] * 1000 for item in raptor_data['raptor']]  # 秒→ミリ秒
    
    # データ抽出
    systems = {
        "Naive RAG": {
            "score": colbert_data['naive_rag']['summary']['avg_score'],
            "top_score": colbert_data['naive_rag']['summary']['top_score'],
            "time_ms": colbert_data['naive_rag']['summary']['avg_time_ms'],
            "std": colbert_data['naive_rag']['summary']['score_std']
        },
        "RAPTOR": {
            "score": np.mean(raptor_scores),
            "top_score": np.mean(raptor_top_scores),
            "time_ms": np.mean(raptor_times),
            "std": np.std(raptor_scores)
        },
        "ColBERT\n(50% sampling)": {
            "score": colbert_data['colbert_rag']['summary']['avg_score'],
            "top_score": colbert_data['colbert_rag']['summary']['top_score'],
            "time_ms": colbert_data['colbert_rag']['summary']['avg_time_ms'],
            "std": colbert_data['colbert_rag']['summary']['score_std']
        }
    }
    
    # Selective RAGの統計計算
    selective_scores = [r['avg_score'] for r in selective_data['results']]
    selective_top_scores = [r['top_score'] for r in selective_data['results']]
    selective_times = [r['time_ms'] for r in selective_data['results']]
    
    systems["Selective RAG\n(Adaptive)"] = {
        "score": np.mean(selective_scores),
        "top_score": np.mean(selective_top_scores),
        "time_ms": np.mean(selective_times),
        "std": np.std(selective_scores)
    }
    
    return systems

def create_comparison_chart(systems):
    """スコアと速度の比較グラフを作成"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    system_names = list(systems.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # ===== 左図: スコア比較 =====
    scores = [systems[name]['score'] for name in system_names]
    top_scores = [systems[name]['top_score'] for name in system_names]
    stds = [systems[name]['std'] for name in system_names]
    
    x = np.arange(len(system_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, scores, width, label='平均スコア', 
                    color=colors, alpha=0.8, yerr=stds, capsize=5)
    bars2 = ax1.bar(x + width/2, top_scores, width, label='トップスコア', 
                    color=colors, alpha=0.5, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('RAGシステム', fontsize=12, fontweight='bold')
    ax1.set_ylabel('検索スコア', fontsize=12, fontweight='bold')
    ax1.set_title('RAGシステム別 検索精度比較', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(system_names, fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1.0)
    
    # スコア値をバーに表示
    for i, (bar, score) in enumerate(zip(bars1, scores)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Naive RAGを基準とした改善率を表示
    baseline = scores[0]
    for i, score in enumerate(scores[1:], 1):
        improvement = ((score - baseline) / baseline) * 100
        color = 'green' if improvement > 0 else 'red'
        ax1.text(i, 0.05, f'{improvement:+.1f}%',
                ha='center', fontsize=9, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # ===== 右図: 速度比較 =====
    times = [systems[name]['time_ms'] for name in system_names]
    
    bars3 = ax2.bar(system_names, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('RAGシステム', fontsize=12, fontweight='bold')
    ax2.set_ylabel('平均応答時間 (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('RAGシステム別 処理速度比較', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticklabels(system_names, fontsize=10)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # 時間値をバーに表示
    for bar, time in zip(bars3, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}ms',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Naive RAGを基準とした速度比を表示
    baseline_time = times[0]
    for i, time in enumerate(times):
        ratio = time / baseline_time
        ax2.text(i, time * 0.05, f'{ratio:.1f}x',
                ha='center', fontsize=9, color='blue', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # 保存
    output_path = Path("output/rag_systems_comparison.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 比較グラフを保存: {output_path}")
    
    return fig

def create_scatter_plot(systems):
    """スコアvs速度の散布図を作成"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    system_names = list(systems.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    markers = ['o', 's', '^', 'D']
    
    for i, name in enumerate(system_names):
        score = systems[name]['score']
        time = systems[name]['time_ms']
        
        ax.scatter(time, score, s=300, color=colors[i], marker=markers[i],
                  alpha=0.7, edgecolors='black', linewidth=2, label=name)
        
        # ラベル表示
        ax.annotate(name, (time, score), xytext=(10, 10),
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))
    
    ax.set_xlabel('平均応答時間 (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均検索スコア', fontsize=12, fontweight='bold')
    ax.set_title('RAGシステム性能マップ: 精度 vs 速度', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 理想領域の強調
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(5, 0.95, '高精度', fontsize=10, color='green', fontweight='bold')
    ax.text(5, 0.55, '低精度', fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    output_path = Path("output/rag_systems_scatter.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 散布図を保存: {output_path}")
    
    return fig

def print_summary_table(systems):
    """サマリーテーブルを表示"""
    print("\n" + "="*80)
    print("RAGシステム性能比較サマリー")
    print("="*80)
    print(f"{'システム':<20} {'平均スコア':>12} {'トップスコア':>12} {'平均時間(ms)':>15} {'標準偏差':>10}")
    print("-"*80)
    
    baseline_score = None
    baseline_time = None
    
    for i, (name, data) in enumerate(systems.items()):
        if i == 0:
            baseline_score = data['score']
            baseline_time = data['time_ms']
            improvement = ""
            time_ratio = ""
        else:
            improvement = f"({((data['score'] - baseline_score) / baseline_score * 100):+.1f}%)"
            time_ratio = f"({data['time_ms'] / baseline_time:.1f}x)"
        
        print(f"{name:<20} {data['score']:>12.4f} {data['top_score']:>12.4f} "
              f"{data['time_ms']:>12.1f} {time_ratio:>3} {data['std']:>10.4f}")
        
        if improvement:
            print(f"{'':>20} {improvement:>12}")
    
    print("="*80)
    print("\n【ベースライン】Naive RAG")
    print("【評価指標】")
    print("  - 平均スコア: 検索結果の平均類似度スコア")
    print("  - トップスコア: 最も関連性の高い結果のスコア")
    print("  - 平均時間: クエリあたりの平均処理時間")
    print("  - 標準偏差: スコアのばらつき")
    print("\n")

def main():
    """メイン処理"""
    print("Loading data...")
    systems = load_data()
    
    print("Creating comparison charts...")
    create_comparison_chart(systems)
    create_scatter_plot(systems)
    
    print_summary_table(systems)
    
    print("\n✓ 完了: すべてのグラフを output/ に保存しました")

if __name__ == "__main__":
    main()
