"""簡易検索テスト"""
import logging
logging.basicConfig(level=logging.DEBUG)

from raptor_rag import RAPTORRAG

r = RAPTORRAG()
r.load('output/raptor_rag.pkl')

results = r.search('河川堤防の設計基準', top_k=5)

print(f"\n検索結果: {len(results)}件")
for i, (content, score, metadata) in enumerate(results):
    print(f"{i+1}. Level={metadata['level']}, Type={metadata['type']}, Score={score:.4f}")
