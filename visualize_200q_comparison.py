"""
200問ベンチマークの4システム比較可視化スクリプト
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 日本語フォント設定
matplotlib.rcParams['font.family'] = 'MS Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

def load_all_results():
    """全システムの結果を読み込み"""
    results = {}
    
    # 1. Selective RAG
    selective_path = Path('selective_rag/output/selective_rag_benchmark_results.json')
    if selective_path.exists():
        with open(selective_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results['Selective RAG'] = {
                'avg_score': data['summary']['overall_avg_score'],
                'avg_time_ms': data['summary']['overall_avg_time_ms'],
                'selection_accuracy': data['summary']['selection_accuracy']
            }
    
    # 2. ColBERT MVP
    colbert_path = Path('colbert_mvp/output/colbert_benchmark_results_200q.json')
    if colbert_path.exists():
        with open(colbert_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results['ColBERT (50%)'] = {
                'avg_score': data['colbert_rag']['summary']['avg_score'],
                'avg_time_ms': data['colbert_rag']['summary']['avg_time_ms']
            }
            results['Naive RAG'] = {
                'avg_score': data['naive_rag']['summary']['avg_score'],
                'avg_time_ms': data['naive_rag']['summary']['avg_time_ms']
            }
    
    # 3. RAPTOR MVP
    raptor_path = Path('raptor_mvp/output/benchmark_results_200q.json')
    if raptor_path.exists():
        with open(raptor_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'raptor' in data:
                raptor_results = data['raptor']
                scores = [r['top_score'] for r in raptor_results]
                times = [r['time'] * 1000 for r in raptor_results]
                results['RAPTOR'] = {
                    'avg_score': sum(scores) / len(scores),
                    'avg_time_ms': sum(times) / len(times)
                }
    
    return results

def create_comparison_charts(results):
    """4システムの比較チャートを作成"""
    
    # システム順序（論理的な順序）
    system_order = ['Naive RAG', 'RAPTOR', 'ColBERT (50%)', 'Selective RAG']
    systems = [s for s in system_order if s in results]
    
    # データ抽出
    scores = [results[s]['avg_score'] for s in systems]
    times = [results[s]['avg_time_ms'] for s in systems]
    
    # カラースキーム
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # Figure作成
    fig = plt.figure(figsize=(15, 10))
    
    # 1. スコア比較（棒グラフ）
    ax1 = plt.subplot(2, 2, 1)
    bars1 = ax1.bar(systems, scores, color=colors[:len(systems)], alpha=0.8, edgecolor='black')
    ax1.set_ylabel('平均スコア', fontsize=12, fontweight='bold')
    ax1.set_title('(a) 精度比較（200問）', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 値をバーに表示
    for bar, score in zip(bars1, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    
    # 2. 速度比較（棒グラフ、対数スケール）
    ax2 = plt.subplot(2, 2, 2)
    bars2 = ax2.bar(systems, times, color=colors[:len(systems)], alpha=0.8, edgecolor='black')
    ax2.set_ylabel('平均応答時間 (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) 速度比較（200問）', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 値をバーに表示
    for bar, time in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    
    # 3. 精度 vs 速度（散布図）
    ax3 = plt.subplot(2, 2, 3)
    
    for i, system in enumerate(systems):
        ax3.scatter(times[i], scores[i], 
                   s=300, c=[colors[i]], alpha=0.7, edgecolor='black', linewidth=2,
                   label=system)
    
    ax3.set_xlabel('平均応答時間 (ms)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('平均スコア', fontsize=12, fontweight='bold')
    ax3.set_title('(c) 精度 vs 速度トレードオフ', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_ylim(0.6, 1.0)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='best', fontsize=9)
    
    # 理想的な領域を示す
    ax3.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5, label='目標精度')
    ax3.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='目標速度')
    
    # 4. システム特性の比較表
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # テーブルデータ作成
    table_data = []
    table_data.append(['システム', 'スコア', '時間(ms)', '特徴'])
    
    characteristics = {
        'Naive RAG': '最速・基本的',
        'RAPTOR': '階層構造',
        'ColBERT (50%)': '最高精度',
        'Selective RAG': '適応的選択'
    }
    
    for system in systems:
        table_data.append([
            system,
            f"{results[system]['avg_score']:.4f}",
            f"{results[system]['avg_time_ms']:.1f}",
            characteristics.get(system, '-')
        ])
    
    # テーブル作成
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # ヘッダー行のスタイル
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    # データ行のスタイル（システムごとに色付け）
    for i, system in enumerate(systems, 1):
        for j in range(4):
            cell = table[(i, j)]
            cell.set_facecolor(colors[i-1])
            cell.set_alpha(0.3)
    
    ax4.set_title('(d) システム特性比較', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('output/comparison_4systems_200q.png', dpi=300, bbox_inches='tight')
    print(f"✓ グラフ保存: output/comparison_4systems_200q.png")
    plt.close()
    
    # Selective RAG詳細グラフ
    if 'Selective RAG' in results and 'selection_accuracy' in results['Selective RAG']:
        create_selective_rag_details(results)

def create_selective_rag_details(results):
    """Selective RAGの詳細分析グラフ"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 選択精度
    selection_acc = results['Selective RAG']['selection_accuracy']
    ax1.bar(['選択精度'], [selection_acc], color='#f39c12', alpha=0.8, edgecolor='black')
    ax1.axhline(y=85, color='red', linestyle='--', label='目標: 85%')
    ax1.set_ylabel('精度 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Selective RAGの選択精度', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 値表示
    ax1.text(0, selection_acc, f'{selection_acc:.1f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 2. パフォーマンス比較（Naive, ColBERT, Selectiveの3つ）
    # 2軸グラフ: 左軸=精度、右軸=速度(ms)
    systems = ['Naive RAG', 'ColBERT (50%)', 'Selective RAG']
    scores = [results[s]['avg_score'] for s in systems]
    times = [results[s]['avg_time_ms'] for s in systems]
    
    x = np.arange(len(systems))
    width = 0.35
    
    # 精度バー（左軸）
    bars1 = ax2.bar(x - width/2, scores, width, label='精度', 
                    color='#2ecc71', alpha=0.8, edgecolor='black')
    ax2.set_ylabel('平均スコア', fontsize=12, fontweight='bold', color='#2ecc71')
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='y', labelcolor='#2ecc71')
    
    # 速度バー（右軸）
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, times, width, label='応答時間', 
                         color='#3498db', alpha=0.8, edgecolor='black')
    ax2_twin.set_ylabel('平均応答時間 (ms)', fontsize=12, fontweight='bold', color='#3498db')
    ax2_twin.set_yscale('log')
    ax2_twin.tick_params(axis='y', labelcolor='#3498db')
    
    ax2.set_title('(b) 精度と速度の実測値比較', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(systems, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # 値表示
    for bar, score in zip(bars1, scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar, time in zip(bars2, times):
        height = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{time:.0f}ms',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/selective_rag_details.png', dpi=300, bbox_inches='tight')
    print(f"✓ グラフ保存: output/selective_rag_details.png")
    plt.close()

def main():
    # 出力ディレクトリ作成
    Path('output').mkdir(exist_ok=True)
    
    # データ読み込み
    results = load_all_results()
    
    print("\n" + "="*60)
    print("200問ベンチマークの可視化")
    print("="*60)
    
    # グラフ作成
    create_comparison_charts(results)
    
    print("\n完了!")

if __name__ == '__main__':
    main()
