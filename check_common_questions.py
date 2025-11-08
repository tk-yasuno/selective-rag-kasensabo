"""
3つのRAGシステムが共通の質問で評価されているか確認
"""

import json
from pathlib import Path

# データファイルパス
RAPTOR_RESULTS = Path("raptor_mvp/output/benchmark_results.json")
COLBERT_RESULTS = Path("colbert_mvp/output/colbert_benchmark_results.json")
SELECTIVE_RESULTS = Path("selective_rag/output/selective_rag_benchmark_results.json")

def load_and_check():
    """各システムの質問データを読み込んで比較"""
    
    # RAPTOR結果
    with open(RAPTOR_RESULTS, 'r', encoding='utf-8') as f:
        raptor_data = json.load(f)
    
    # ColBERT結果
    with open(COLBERT_RESULTS, 'r', encoding='utf-8') as f:
        colbert_data = json.load(f)
    
    # Selective RAG結果
    with open(SELECTIVE_RESULTS, 'r', encoding='utf-8') as f:
        selective_data = json.load(f)
    
    print("="*80)
    print("RAGシステム別 質問数")
    print("="*80)
    
    # 質問数確認
    raptor_naive_count = len(raptor_data['naive'])
    raptor_raptor_count = len(raptor_data['raptor'])
    colbert_naive_count = colbert_data['naive_rag']['summary']['total_questions']
    colbert_colbert_count = colbert_data['colbert_rag']['summary']['total_questions']
    selective_count = selective_data['metadata']['total_questions']
    
    print(f"\n【RAPTOR MVPプロジェクト】")
    print(f"  - Naive RAG: {raptor_naive_count}問")
    print(f"  - RAPTOR: {raptor_raptor_count}問")
    
    print(f"\n【ColBERT MVPプロジェクト】")
    print(f"  - Naive RAG: {colbert_naive_count}問")
    print(f"  - ColBERT RAG: {colbert_colbert_count}問")
    
    print(f"\n【Selective RAGプロジェクト】")
    print(f"  - Selective RAG: {selective_count}問")
    print(f"    * Fine-grained (細粒度): 100問")
    print(f"    * Coarse-grained (粗粒度): 100問")
    
    print("\n" + "="*80)
    print("質問内容の比較")
    print("="*80)
    
    # 質問内容取得
    raptor_questions = [q['question'] for q in raptor_data['naive']]
    colbert_questions = [q['question'] for q in colbert_data['naive_rag']['details']]
    selective_questions = [q['question'] for q in selective_data['results']]
    
    # RAPTOR vs ColBERT
    raptor_set = set(raptor_questions)
    colbert_set = set(colbert_questions)
    selective_set = set(selective_questions)
    
    common_raptor_colbert = raptor_set & colbert_set
    common_all = raptor_set & colbert_set & selective_set
    
    print(f"\n【質問の共通性】")
    print(f"  RAPTOR ∩ ColBERT: {len(common_raptor_colbert)}問が共通")
    print(f"  RAPTOR ∩ ColBERT ∩ Selective: {len(common_all)}問が共通")
    
    print(f"\n【質問の重複状況】")
    if len(common_raptor_colbert) == 100:
        print(f"  ✓ RAPTORとColBERTは同じ100問を使用")
    else:
        print(f"  ⚠ RAPTORとColBERTで異なる質問セット")
    
    if len(common_all) == 0:
        print(f"  ⚠ Selective RAGは独自の200問を使用（RAPTOR/ColBERTと異なる）")
    else:
        print(f"  ✓ 3システム間で{len(common_all)}問が共通")
    
    # 質問例の表示
    print("\n" + "="*80)
    print("質問例の比較")
    print("="*80)
    
    print(f"\n【RAPTOR/ColBERT - 最初の5問】")
    for i, q in enumerate(raptor_questions[:5], 1):
        print(f"  {i}. {q}")
    
    print(f"\n【Selective RAG - 最初の5問】")
    for i, q in enumerate(selective_questions[:5], 1):
        print(f"  {i}. {q}")
    
    # 粒度分析（Selective RAG）
    print("\n" + "="*80)
    print("Selective RAG 質問粒度分布")
    print("="*80)
    
    fine_count = sum(1 for r in selective_data['results'] if r['true_granularity'] == 'fine')
    coarse_count = sum(1 for r in selective_data['results'] if r['true_granularity'] == 'coarse')
    
    print(f"\n  Fine-grained (細粒度): {fine_count}問")
    print(f"  Coarse-grained (粗粒度): {coarse_count}問")
    
    # システム選択分布
    naive_selected = sum(1 for r in selective_data['results'] if r['selected_system'] == 'naive')
    colbert_selected = sum(1 for r in selective_data['results'] if r['selected_system'] == 'colbert')
    
    print(f"\n【システム選択分布】")
    print(f"  Naive RAG選択: {naive_selected}問 ({naive_selected/selective_count*100:.1f}%)")
    print(f"  ColBERT選択: {colbert_selected}問 ({colbert_selected/selective_count*100:.1f}%)")
    
    print("\n" + "="*80)
    print("結論")
    print("="*80)
    print(f"""
【異なる評価セット】
• RAPTOR & ColBERT MVP: 同じ100問を使用
• Selective RAG: 独自の200問を使用（100問細粒度 + 100問粗粒度）

【評価の違い】
• RAPTOR/ColBERT: 一般的な河川砂防技術の質問100問
• Selective RAG: 専門性の粒度で分類された200問
  - 細粒度: 具体的数値・規格・計算式など
  - 粗粒度: 概念・定義・目的・比較など

【比較の妥当性】
⚠ 異なる質問セットのため、直接的なスコア比較は注意が必要
✓ ただし、システムの特性（精度vs速度）の傾向は把握可能
    """)

if __name__ == "__main__":
    load_and_check()
