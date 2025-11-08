# 河川砂防RAPTOR MVP

河川砂防技術基準を対象とした、Naive RAGとRAPTORの性能比較MVPシステム

## 📋 概要

このプロジェクトは、河川砂防ダム技術基準の文書検索において、シンプルなベクトル検索（Naive RAG）と階層的検索（RAPTOR）の性能を比較するための最小実装です。

## 🏗 構成

```
raptor_mvp/
├── config.py              # 設定ファイル
├── naive_rag.py           # Naive RAG実装（FAISSベクトル検索）
├── raptor_rag.py          # RAPTOR実装（階層的クラスタリング）
├── benchmark.py           # ベンチマーク比較
├── visualize.py           # 可視化
├── main.py                # メイン実行スクリプト
├── output/
│   ├── benchmark_questions_100.json  # 100問の評価質問
│   ├── naive_rag.pkl                 # Naive RAGキャッシュ
│   ├── raptor_rag.pkl                # RAPTORキャッシュ
│   ├── benchmark_results.json        # ベンチマーク結果
│   ├── raptor_tree.png               # ツリー構造可視化
│   ├── raptor_levels.png             # レベル分布
│   ├── benchmark_comparison.png      # 性能比較
│   └── benchmark_categories.png      # カテゴリ別性能
└── README.md
```

## 🚀 使い方

### 1. RAGシステム構築

```bash
python main.py build
```

- Naive RAG: 文書をチャンク化してFAISSインデックス構築
- RAPTOR: 階層的クラスタリングでツリー構造構築

### 2. ベンチマーク実行

```bash
# 全100問で実行
python main.py benchmark

# サンプル10問で実行
python main.py benchmark --sample 10
```

### 3. 可視化

```bash
python main.py visualize
```

生成されるグラフ：
- `raptor_tree.png`: ツリー構造
- `raptor_levels.png`: レベル別ノード分布
- `benchmark_comparison.png`: Naive vs RAPTOR比較
- `benchmark_categories.png`: カテゴリ別性能

### 4. クエリテスト

```bash
# デフォルトクエリでテスト
python main.py test

# カスタムクエリでテスト
python main.py test --query "砂防堰堤の設計基準は？"
```

### 5. 一括実行

```bash
# 構築 + ベンチマーク + 可視化
python main.py all --sample 10
```

## 📊 評価指標

### 比較項目

1. **応答時間**: クエリ実行時間（ミリ秒）
2. **検索精度**: Top-K類似度スコア
3. **一貫性**: カテゴリ別の安定性

### 100問の質問カテゴリ

1. 河川施設・構造物（15問）
2. 水文・水理・流域管理（15問）
3. 砂防・土砂災害（15問）
4. ダム構造・管理（15問）
5. 地すべり・急傾斜地・雪崩（15問）
6. 点検・健全度・補修（15問）
7. 設計・環境・抽象評価（10問）

## ⚙️ 設定

`config.py`で以下を調整可能：

```python
# モデル設定
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG設定
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 5

# RAPTOR設定
MAX_TREE_DEPTH = 3
MIN_CLUSTER_SIZE = 3
MAX_CLUSTERS = 5

# クラスタリング評価戦略
CLUSTERING_STRATEGY = "balanced"  # silhouette, dbi, balanced
```

## 📈 期待される結果

### Naive RAGの特徴
- ✅ 高速（シンプルなベクトル検索）
- ❌ 文脈の断片化
- ❌ 類似語による誤検索

### RAPTORの特徴
- ✅ 階層構造による文脈保持
- ✅ クラスタリングによる関連情報の集約
- ✅ 再ランキングによる精度向上
- ❌ 構築時間がやや長い

## 🔧 必要なパッケージ

```bash
pip install sentence-transformers faiss-cpu scikit-learn networkx matplotlib tqdm
```

GPU版FAISSを使用する場合：
```bash
pip install faiss-gpu
```

## 📝 データ

**知識ベース**: `C:\Users\yasun\LangChain\learning-langchain\kasensabo-raptor\data\kasensabo_knowledge_base`

8個のMarkdownファイル：
- 00_training_overview_2025.md
- 01_training_chousa_2025.md
- 02_training_keikaku_kihon_2025.md
- 03_training_keikaku_shisetsu_2025.md
- 04_training_sekkei_2025.md
- 05_training_ijikanri_kasen_2025.md
- 06_training_ijikanri_dam_2025.md
- 07_training_ijikanri_sabo_2025.md

**語彙辞書**: `kasensabo_vocab.py`（200+専門用語）

## 📄 ライセンス

河川砂防技術基準に基づく研究・開発用

## 🤝 参考

- RAPTOR論文: "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
- ベースコード: `base-code-enhanced-treg-raptor/true_raptor_builder.py`
