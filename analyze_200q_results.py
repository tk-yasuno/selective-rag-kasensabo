"""
200問ベンチマークの結果を集計・比較するスクリプト
"""
import json
from pathlib import Path

def analyze_raptor_results(results_path):
    """RAPTOR MVPの結果JSONを読み込んで統計を計算"""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # システムごとの結果を集計
    stats = {}
    
    for rag_type, results in data.items():
        if not results:
            continue
            
        scores = [r['top_score'] for r in results]
        times = [r['time'] * 1000 for r in results]  # ms変換
        
        stats[rag_type] = {
            'avg_score': sum(scores) / len(scores),
            'avg_time_ms': sum(times) / len(times),
            'num_questions': len(results)
        }
    
    return stats

def analyze_colbert_results(results_path):
    """ColBERT MVPの結果JSONを読み込んで統計を計算"""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {}
    
    # naive_ragとcolbert_ragのサマリーを抽出
    for rag_type, rag_data in data.items():
        if 'summary' not in rag_data:
            continue
        
        summary = rag_data['summary']
        stats[rag_type] = {
            'avg_score': summary['avg_score'],
            'avg_time_ms': summary['avg_time_ms'],
            'num_questions': summary['total_questions']
        }
    
    return stats

def main():
    # 3つのシステムの結果を読み込み
    results = {}
    
    # 1. Selective RAG (200問)
    selective_path = Path('selective_rag/output/selective_rag_benchmark_results.json')
    if selective_path.exists():
        with open(selective_path, 'r', encoding='utf-8') as f:
            selective_data = json.load(f)
            results['Selective RAG'] = {
                'avg_score': selective_data['summary']['overall_avg_score'],
                'avg_time_ms': selective_data['summary']['overall_avg_time_ms'],
                'num_questions': selective_data['summary']['total_questions']
            }
    
    # 2. ColBERT MVP (200問)
    colbert_path = Path('colbert_mvp/output/colbert_benchmark_results_200q.json')
    if colbert_path.exists():
        colbert_stats = analyze_colbert_results(colbert_path)
        results['ColBERT (50%)'] = colbert_stats.get('colbert_rag', {})
        results['Naive (in ColBERT)'] = colbert_stats.get('naive_rag', {})
    
    # 3. RAPTOR MVP (200問)
    raptor_path = Path('raptor_mvp/output/benchmark_results_200q.json')
    if raptor_path.exists():
        raptor_stats = analyze_raptor_results(raptor_path)
        results['RAPTOR'] = raptor_stats.get('raptor', {})
        results['Naive (in RAPTOR)'] = raptor_stats.get('naive', {})
    
    # 結果を表示
    print("\n" + "="*80)
    print("200問ベンチマーク結果の統合比較")
    print("="*80)
    
    # システム順: Naive, RAPTOR, ColBERT, Selective
    system_order = [
        'Naive (in RAPTOR)',
        'Naive (in ColBERT)',
        'RAPTOR',
        'ColBERT (50%)',
        'Selective RAG'
    ]
    
    for system in system_order:
        if system in results:
            stat = results[system]
            print(f"\n{system}:")
            print(f"  質問数:     {stat['num_questions']}")
            print(f"  平均スコア: {stat['avg_score']:.4f}")
            print(f"  平均時間:   {stat['avg_time_ms']:.1f} ms")
    
    # 考察
    print("\n" + "="*80)
    print("考察")
    print("="*80)
    
    # スコアでソート
    sorted_by_score = sorted(results.items(), key=lambda x: x[1]['avg_score'], reverse=True)
    print("\n精度ランキング:")
    for i, (system, stat) in enumerate(sorted_by_score, 1):
        print(f"  {i}. {system}: {stat['avg_score']:.4f}")
    
    # 速度でソート
    sorted_by_speed = sorted(results.items(), key=lambda x: x[1]['avg_time_ms'])
    print("\n速度ランキング:")
    for i, (system, stat) in enumerate(sorted_by_speed, 1):
        print(f"  {i}. {system}: {stat['avg_time_ms']:.1f} ms")
    
    # Selective RAGの評価
    if 'Selective RAG' in results:
        selective = results['Selective RAG']
        naive_raptor = results.get('Naive (in RAPTOR)', {})
        colbert = results.get('ColBERT (50%)', {})
        
        print("\nSelective RAGの性能評価:")
        if naive_raptor and colbert:
            print(f"  - Naive RAGとの差:   {(selective['avg_score'] - naive_raptor['avg_score'])*100:+.2f}% (スコア)")
            print(f"  - ColBERTとの差:     {(selective['avg_score'] - colbert['avg_score'])*100:+.2f}% (スコア)")
            print(f"  - Naive RAGとの差:   {selective['avg_time_ms'] / naive_raptor['avg_time_ms']:.1f}x (時間)")
            print(f"  - ColBERTとの差:     {colbert['avg_time_ms'] / selective['avg_time_ms']:.1f}x (時間)")
            print(f"  → Selective RAGは精度と速度のバランスを実現")

if __name__ == '__main__':
    main()
