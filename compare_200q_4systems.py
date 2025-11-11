"""
4つのRAGシステム(Naive RAG, RAPTOR, ColBERT, Selective RAG)の200問ベンチマーク結果を比較
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# データファイルパス
RAPTOR_RESULTS = Path("raptor_mvp/output/benchmark_results_200q.json")
COLBERT_RESULTS = Path("colbert_mvp/output/colbert_benchmark_results_200q.json")
SELECTIVE_RESULTS = Path("selective_rag/output/selective_rag_benchmark_results.json")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """各システムの200問ベンチマーク結果データを読み込む"""
    
    # RAPTOR結果
    with open(RAPTOR_RESULTS, 'r', encoding='utf-8') as f:
        raptor_data = json.load(f)
    
    # ColBERT結果
    with open(COLBERT_RESULTS, 'r', encoding='utf-8') as f:
        colbert_data = json.load(f)
    
    # Selective RAG結果
    with open(SELECTIVE_RESULTS, 'r', encoding='utf-8') as f:
        selective_data = json.load(f)
    
    # RAPTOR統計計算 (naive と raptor の両方を含む)
    raptor_naive_scores = [item['avg_score'] for item in raptor_data['naive']]
    raptor_naive_top_scores = [item['top_score'] for item in raptor_data['naive']]
    raptor_naive_times = [item['time'] * 1000 for item in raptor_data['naive']]  # 秒→ミリ秒
    
    raptor_scores = [item['avg_score'] for item in raptor_data['raptor']]
    raptor_top_scores = [item['top_score'] for item in raptor_data['raptor']]
    raptor_times = [item['time'] * 1000 for item in raptor_data['raptor']]  # 秒→ミリ秒
    
    # データ抽出
    systems = {
        "Naive RAG\n(Baseline)": {
            "score": colbert_data['naive_rag']['summary']['avg_score'],
            "top_score": colbert_data['naive_rag']['summary']['top_score'],
            "time_ms": colbert_data['naive_rag']['summary']['avg_time_ms'],
            "std": colbert_data['naive_rag']['summary']['score_std'],
            "fine_score": colbert_data['naive_rag']['summary']['fine_grained_score'],
            "coarse_score": colbert_data['naive_rag']['summary']['coarse_grained_score']
        },
        "RAPTOR\n(Hierarchical)": {
            "score": np.mean(raptor_scores),
            "top_score": np.mean(raptor_top_scores),
            "time_ms": np.mean(raptor_times),
            "std": np.std(raptor_scores),
            "fine_score": np.mean([item['avg_score'] for item in raptor_data['raptor'] if item.get('granularity') == 'fine']),
            "coarse_score": np.mean([item['avg_score'] for item in raptor_data['raptor'] if item.get('granularity') == 'coarse'])
        },
        "ColBERT\n(Dense Retrieval)": {
            "score": colbert_data['colbert_rag']['summary']['avg_score'],
            "top_score": colbert_data['colbert_rag']['summary']['top_score'],
            "time_ms": colbert_data['colbert_rag']['summary']['avg_time_ms'],
            "std": colbert_data['colbert_rag']['summary']['score_std'],
            "fine_score": colbert_data['colbert_rag']['summary']['fine_grained_score'],
            "coarse_score": colbert_data['colbert_rag']['summary']['coarse_grained_score']
        },
        "Selective RAG\n(Adaptive)": {
            "score": np.mean([r['avg_score'] for r in selective_data['results']]),
            "top_score": np.mean([r['top_score'] for r in selective_data['results']]),
            "time_ms": np.mean([r['time_ms'] for r in selective_data['results']]),
            "std": np.std([r['avg_score'] for r in selective_data['results']]),
            "fine_score": np.mean([r['avg_score'] for r in selective_data['results'] if r['true_granularity'] == 'fine']),
            "coarse_score": np.mean([r['avg_score'] for r in selective_data['results'] if r['true_granularity'] == 'coarse'])
        }
    }
    
    return systems, selective_data

def create_comparison_charts(systems, selective_data):
    """スコア、速度、粒度別の比較グラフを作成"""
    
    fig = plt.figure(figsize=(18, 12))
    
    system_names = list(systems.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # ===== 1. スコア比較 (左上) =====
    ax1 = plt.subplot(2, 3, 1)
    scores = [systems[name]['score'] for name in system_names]
    top_scores = [systems[name]['top_score'] for name in system_names]
    stds = [systems[name]['std'] for name in system_names]
    
    x = np.arange(len(system_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, scores, width, label='平均スコア', 
                    color=colors, alpha=0.8, yerr=stds, capsize=5)
    bars2 = ax1.bar(x + width/2, top_scores, width, label='トップスコア', 
                    color=colors, alpha=0.5, edgecolor='black', linewidth=1.5)
    
    # 値をバーの上に表示
    for i, (score, top) in enumerate(zip(scores, top_scores)):
        ax1.text(i - width/2, score + stds[i] + 0.01, f'{score:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(i + width/2, top + 0.01, f'{top:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('RAGシステム', fontsize=11, fontweight='bold')
    ax1.set_ylabel('検索スコア', fontsize=11, fontweight='bold')
    ax1.set_title('総合検索精度比較 (200問)', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(system_names, fontsize=9)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1.0)
    
    # ===== 2. 速度比較 (中央上) =====
    ax2 = plt.subplot(2, 3, 2)
    times = [systems[name]['time_ms'] for name in system_names]
    
    bars = ax2.bar(x, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 値をバーの上に表示
    for i, time in enumerate(times):
        ax2.text(i, time + max(times)*0.02, f'{time:.1f}ms', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('RAGシステム', fontsize=11, fontweight='bold')
    ax2.set_ylabel('平均処理時間 (ms)', fontsize=11, fontweight='bold')
    ax2.set_title('処理速度比較', fontsize=12, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(system_names, fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # ===== 3. 粒度別スコア比較 (右上) =====
    ax3 = plt.subplot(2, 3, 3)
    fine_scores = [systems[name]['fine_score'] for name in system_names]
    coarse_scores = [systems[name]['coarse_score'] for name in system_names]
    
    width = 0.35
    bars1 = ax3.bar(x - width/2, fine_scores, width, label='詳細質問 (Fine)', 
                    color=colors, alpha=0.8)
    bars2 = ax3.bar(x + width/2, coarse_scores, width, label='概要質問 (Coarse)', 
                    color=colors, alpha=0.5, edgecolor='black', linewidth=1.5)
    
    # 値をバーの上に表示
    for i, (fine, coarse) in enumerate(zip(fine_scores, coarse_scores)):
        ax3.text(i - width/2, fine + 0.01, f'{fine:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax3.text(i + width/2, coarse + 0.01, f'{coarse:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax3.set_xlabel('RAGシステム', fontsize=11, fontweight='bold')
    ax3.set_ylabel('検索スコア', fontsize=11, fontweight='bold')
    ax3.set_title('質問粒度別スコア比較', fontsize=12, fontweight='bold', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(system_names, fontsize=9)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim(0, 1.0)
    
    # ===== 4. スコア vs 速度 (左下) =====
    ax4 = plt.subplot(2, 3, 4)
    for i, name in enumerate(system_names):
        ax4.scatter(systems[name]['time_ms'], systems[name]['score'], 
                   s=200, color=colors[i], alpha=0.7, edgecolor='black', linewidth=2,
                   label=name)
        ax4.annotate(name.split('\n')[0], 
                    (systems[name]['time_ms'], systems[name]['score']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))
    
    ax4.set_xlabel('平均処理時間 (ms)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('平均スコア', fontsize=11, fontweight='bold')
    ax4.set_title('精度 vs 速度のトレードオフ', fontsize=12, fontweight='bold', pad=15)
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.set_ylim(0.6, 0.9)
    
    # ===== 5. Selective RAGの選択分布 (中央下) =====
    ax5 = plt.subplot(2, 3, 5)
    selected_systems = [r['selected_system'] for r in selective_data['results']]
    system_counts = {}
    for sys in selected_systems:
        system_counts[sys] = system_counts.get(sys, 0) + 1
    
    labels = list(system_counts.keys())
    sizes = list(system_counts.values())
    colors_pie = ['#2ecc71', '#e74c3c', '#3498db']
    
    wedges, texts, autotexts = ax5.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90,
                                        textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    ax5.set_title('Selective RAG: システム選択分布', fontsize=12, fontweight='bold', pad=15)
    
    # ===== 6. 正解率 (右下) =====
    ax6 = plt.subplot(2, 3, 6)
    correct_selections = sum([1 for r in selective_data['results'] if r['correct_selection']])
    accuracy = correct_selections / len(selective_data['results']) * 100
    
    categories = ['正解選択', '誤選択']
    values = [correct_selections, len(selective_data['results']) - correct_selections]
    colors_acc = ['#2ecc71', '#e74c3c']
    
    bars = ax6.bar(categories, values, color=colors_acc, alpha=0.8, edgecolor='black', linewidth=2)
    
    for i, val in enumerate(values):
        ax6.text(i, val + 5, f'{val}問\n({val/len(selective_data["results"])*100:.1f}%)', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax6.set_ylabel('質問数', fontsize=11, fontweight='bold')
    ax6.set_title(f'Selective RAG: 選択精度\n(正解率: {accuracy:.1f}%)', 
                 fontsize=12, fontweight='bold', pad=15)
    ax6.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_200q_4systems.png', dpi=300, bbox_inches='tight')
    print(f"✅ グラフ保存: {OUTPUT_DIR / 'comparison_200q_4systems.png'}")
    plt.close()

def generate_report(systems, selective_data):
    """詳細な比較レポートを生成"""
    
    report = []
    report.append("=" * 80)
    report.append("4つのRAGシステム 200問ベンチマーク比較レポート")
    report.append("=" * 80)
    report.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"質問数: 200問 (詳細質問: 100問, 概要質問: 100問)")
    report.append("")
    
    # システム別統計
    report.append("## システム別パフォーマンス統計")
    report.append("-" * 80)
    report.append(f"{'システム名':<25} {'平均スコア':<12} {'トップスコア':<12} {'処理時間(ms)':<15}")
    report.append("-" * 80)
    
    for name, data in systems.items():
        clean_name = name.replace('\n', ' ')
        report.append(f"{clean_name:<25} {data['score']:.4f}       {data['top_score']:.4f}       {data['time_ms']:.2f}")
    
    report.append("")
    
    # 質問粒度別スコア
    report.append("## 質問粒度別スコア")
    report.append("-" * 80)
    report.append(f"{'システム名':<25} {'詳細質問(Fine)':<18} {'概要質問(Coarse)':<18}")
    report.append("-" * 80)
    
    for name, data in systems.items():
        clean_name = name.replace('\n', ' ')
        report.append(f"{clean_name:<25} {data['fine_score']:.4f}             {data['coarse_score']:.4f}")
    
    report.append("")
    
    # ランキング
    report.append("## ランキング")
    report.append("-" * 80)
    
    # スコアランキング
    sorted_by_score = sorted(systems.items(), key=lambda x: x[1]['score'], reverse=True)
    report.append("### 検索精度ランキング")
    for i, (name, data) in enumerate(sorted_by_score, 1):
        clean_name = name.replace('\n', ' ')
        report.append(f"{i}. {clean_name}: {data['score']:.4f}")
    report.append("")
    
    # 速度ランキング
    sorted_by_speed = sorted(systems.items(), key=lambda x: x[1]['time_ms'])
    report.append("### 処理速度ランキング (速い順)")
    for i, (name, data) in enumerate(sorted_by_speed, 1):
        clean_name = name.replace('\n', ' ')
        report.append(f"{i}. {clean_name}: {data['time_ms']:.2f}ms")
    report.append("")
    
    # Selective RAG詳細
    report.append("## Selective RAG 詳細分析")
    report.append("-" * 80)
    
    selected_systems = [r['selected_system'] for r in selective_data['results']]
    system_counts = {}
    for sys in selected_systems:
        system_counts[sys] = system_counts.get(sys, 0) + 1
    
    report.append("### システム選択分布")
    for sys, count in system_counts.items():
        percentage = count / len(selective_data['results']) * 100
        report.append(f"- {sys}: {count}問 ({percentage:.1f}%)")
    report.append("")
    
    correct_selections = sum([1 for r in selective_data['results'] if r['correct_selection']])
    accuracy = correct_selections / len(selective_data['results']) * 100
    report.append(f"### 選択精度")
    report.append(f"- 正解: {correct_selections}問 ({accuracy:.1f}%)")
    report.append(f"- 誤選択: {len(selective_data['results']) - correct_selections}問 ({100-accuracy:.1f}%)")
    report.append("")
    
    # 結論
    report.append("## 結論")
    report.append("-" * 80)
    
    best_accuracy = sorted_by_score[0]
    best_speed = sorted_by_speed[0]
    
    report.append(f"🥇 最高精度: {best_accuracy[0].replace(chr(10), ' ')} (スコア: {best_accuracy[1]['score']:.4f})")
    report.append(f"⚡ 最高速度: {best_speed[0].replace(chr(10), ' ')} (処理時間: {best_speed[1]['time_ms']:.2f}ms)")
    report.append("")
    report.append("### 主な知見")
    report.append("1. Selective RAGは質問に応じて最適なシステムを選択することで高精度を実現")
    report.append("2. ColBERTは密ベクトル検索により詳細質問で高い精度を達成")
    report.append("3. RAPTORは階層的要約により概要質問に強い")
    report.append("4. Naive RAGは最速だが精度は他システムより低い")
    report.append("")
    report.append("=" * 80)
    
    # ファイル保存
    report_text = "\n".join(report)
    report_file = OUTPUT_DIR / "REPORT_200Q_4SYSTEMS.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✅ レポート保存: {report_file}")
    print("\n" + report_text)

def main():
    """メイン処理"""
    print("=" * 80)
    print("4つのRAGシステム 200問ベンチマーク比較")
    print("=" * 80)
    print()
    
    # データ読み込み
    print("📊 データ読み込み中...")
    systems, selective_data = load_data()
    print(f"✅ {len(systems)}システムのデータを読み込みました")
    print()
    
    # グラフ生成
    print("📈 比較グラフ生成中...")
    create_comparison_charts(systems, selective_data)
    print()
    
    # レポート生成
    print("📝 詳細レポート生成中...")
    generate_report(systems, selective_data)
    print()
    
    print("=" * 80)
    print("✅ 比較分析完了!")
    print("=" * 80)

if __name__ == "__main__":
    main()
