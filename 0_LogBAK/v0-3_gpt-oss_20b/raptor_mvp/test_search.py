"""RAPTORの検索戦略をテスト"""
from raptor_rag import RAPTORRAG

# RAPTORをロード
r = RAPTORRAG()
r.load('output/raptor_rag.pkl')

# テストクエリ
queries = [
    '河川堤防の設計基準は何ですか',
    '砂防ダムのコンクリート強度基準は',
    '地すべり対策の調査方法は'
]

for query in queries:
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print('='*80)
    
    results = r.search(query, top_k=5)
    
    print(f"\nSearch results ({len(results)} items):\n")
    
    for i, (content, score, metadata) in enumerate(results):
        level = metadata.get('level', '?')
        node_type = metadata.get('type', 'unknown')
        node_id = metadata.get('node_id', '?')
        
        print(f"{i+1}. Level={level}, Type={node_type}, Score={score:.4f}")
        print(f"   NodeID: {node_id}")
        print(f"   Content: {content[:150]}...")
        print()
