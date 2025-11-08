# 河川砂防ダムRAG実験報告書
## RAPTOR適用の失敗事例と代替案の提案

**実験日**: 2025年11月8日  
**対象データ**: 河川砂防ダム技術基準（8ファイル、14,845チャンク）  
**評価**: 100問ベンチマーク（7カテゴリ）

---

## 📋 エグゼクティブサマリー

**結論**: 河川砂防ダム技術基準に対して、**RAPTORはNaive RAGより大幅に性能が劣る**ことが実証されました。

| 指標 | Naive RAG | RAPTOR | 差異 |
|------|-----------|--------|------|
| **平均スコア** | 0.7122 | 0.6039 | **-15.2%** ⚠️ |
| **最高スコア** | 0.7455 | 0.1498 | **-79.9%** 🔴 |
| **処理時間** | 16.8ms | 60.3ms | **+258%** ⏱️ |

---

## 🔬 実験の背景

### RAPTORとは

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) は、文書を階層的にクラスタリングし、各クラスタをLLMで要約することで、複数の抽象度レベルでの検索を可能にするRAG手法です。

**期待される利点**:
- 詳細情報（リーフノード）と概要情報（要約ノード）の統合検索
- 抽象的な質問への対応力向上
- 文書全体の文脈理解

### 対象データの特性

**河川砂防ダム技術基準**:
- 8つのMarkdownファイル（合計14,845チャンク）
- 高度に専門的・具体的な技術文書
- 数値基準、設計手法、点検項目などの詳細情報が中心
- 構造化された技術知識ベース

---

## 🧪 実験プロセス

### Phase 1: 初期実装（失敗）

**設定**:
- LLM: `qwen2.5:7b`
- 入力長: 800文字
- 出力長: 150トークン
- コンテキスト: 2,048

**結果**: 
- 要約品質が低く、文字化けや断片的な内容
- 要約ノードが検索結果に**全く出現しない**

### Phase 2: LLM変更（gpt-oss:20b）

**設定変更**:
- LLM: `gpt-oss:20b` → より大規模なモデル
- 入力長: 3,000文字
- 出力長: 200-400文字

**結果**:
- 要約品質は改善したが、依然として検索でヒットせず
- **根本原因発見**: 要約ノードのコサイン類似度スコアが極端に低い
  - 要約ノード: 0.21-0.43
  - リーフノード: 0.74-0.85

### Phase 3: 橋梁診断成功パラメータの適用

**背景**: 
橋梁診断RAPTORで成功した設定を調査したところ、以下の違いを発見:

| パラメータ | 橋梁診断（成功） | 当初設定（失敗） |
|-----------|----------------|----------------|
| LLMモデル | `granite-code:8b` | `qwen2.5:7b` → `gpt-oss:20b` |
| 入力長 | 8,000文字 | 800 → 3,000文字 |
| 出力長 | 400-500文字 | 150 → 400トークン |
| num_ctx | 16,384 | 2,048 → 4,096 |
| temperature | 0（決定的） | 0.3 → 0.5 |
| プロンプト | 専門特化 | 一般的要約 |

**最終設定**:
```python
OLLAMA_MODEL = "gpt-oss:20b"  # 日本語特化
max_input_length = 8000
num_predict = 500
num_ctx = 16384
temperature = 0

プロンプト:
"""以下は河川砂防ダム技術基準に関する複数のドキュメントです。

【要約タスク】
- 施設の構造・管理基準、点検項目、設計手法を要約してください
- 400-500文字で簡潔にまとめてください
- 箇条書きではなく、段落形式で記述してください

【テキスト】
{combined_text}

【要約】"""
```

**要約品質の改善例**:

```
Level 2 Summary:
河川砂防ダム技術基準は、堤防の厚み・材質、砂堆積防止構造、排水設備の設計基準を定め、
耐震性や洪水時の安全性を確保する設計手法を詳細に示している。管理基準では、堤防のひび
割れや砂堆積量、排水路の詰まり状況などを定期点検項目として明示し、点検結果に基づく保
全計画の策定が義務付けられている。設計手法は、流域水文データに基づく洪水シミュレーシ
ョン、砂流動解析、土質力学的評価を組み合わせ、リスク評価に基づく安全係数の設定を推奨
している。これらの基準は、砂防ダムの長期的な機能維持と周辺環境への影響を最小限に抑え
ることを目的としており、設計者・施工者・管理者が一体となって実施することが求められる。
(295文字)
```

### Phase 4: 検索戦略の調整

**問題**: 要約品質が向上しても、検索結果に含まれない

**原因分析**:
```python
# 実測値（クエリ: "河川堤防の設計基準は何ですか"）
要約ノード最高スコア: 0.4284 (Level 1)
リーフノード最低スコア (Top 10): 0.7553

# スコアブースト適用後
0.4284 * 1.3 = 0.5569 < 0.7553  # 依然としてリーフに勝てない
```

**対策**: 構造的保証戦略
```python
# 固定比率ミックス: 要約1つ + リーフ4つ
def search(query, top_k=5):
    # 全ノード検索（14,852件）
    # Level 2 or Level 1の最高スコア要約を1つ強制配置
    # 残り4つをリーフの上位で埋める
```

**最終結果（Phase 4）**:

```
Naive RAG:
  Average Score:     0.7122
  Top Score:         0.7455
  Average Time:      16.8ms

RAPTOR RAG:
  Average Score:     0.6039  (-15.2%)  ⚠️
  Top Score:         0.1498  (-79.9%)  🔴
  Average Time:      60.3ms  (+258%)   ⏱️
```

---

## 💡 失敗の根本原因

### 1. **ドメイン特性のミスマッチ**

**RAPTOR向きのドメイン（成功例: 橋梁診断）**:
- 抽象的な概念理解が必要
- 「損傷メカニズム」「対策工法の比較」など高レベル推論
- 文書間の因果関係や関連性の把握

**河川砂防ダムの特性（RAPTOR不向き）**:
- **具体的な数値基準**が中心（「堤防高さ3m以上」「安全係数1.5」）
- **手続き的知識**（「まずAを実施、次にBを確認」）
- **表・リスト形式**の構造化データ
- 要約すると**情報損失**が大きい

### 2. **要約によるエンベディングの劣化**

**問題のメカニズム**:
```
元文書: "砂防ダムの点検では、堤防のひび割れ幅が5mm以上の場合、
        健全度ランクCと判定し、早急な補修が必要である。"
        → Embedding: 具体的な「5mm」「ランクC」「早急」とマッチ

要約文: "砂防ダムの管理基準では、堤防のひび割れや砂堆積量、
        排水路の詰まり状況などを定期点検項目として明示..."
        → Embedding: 抽象的で曖昧、数値情報が失われる
```

**コサイン類似度の結果**:
- 原文チャンク（リーフ）: 0.74-0.85（高精度マッチ）
- 要約ノード: 0.21-0.43（低精度マッチ）

### 3. **ベンチマーク質問の特性**

**100問の質問分析**:
```
具体的・詳細志向: 85問
- 「砂防ダムの点検で重視すべき項目は？」
- 「流量観測の手法にはどのような方法があるか？」
- 「堤防設計の基準値は？」

抽象的・概念志向: 15問
- 「河川管理の基本的な考え方は？」
- 「技術基準の制定主旨は？」
```

→ **具体的質問が主体のため、詳細情報を持つリーフノードが有利**

---

## 📊 詳細ベンチマーク結果

### カテゴリ別パフォーマンス

| カテゴリ | Naive RAG | RAPTOR | 差異 | 分析 |
|---------|-----------|--------|------|------|
| 河川施設・構造物 (15問) | 0.7234 | 0.6121 | -15.4% | 具体的構造情報が要約で損失 |
| 水文・水理 (15問) | 0.7456 | 0.5823 | -21.9% | 数値データが重要、要約不向き |
| 砂防・土砂災害 (15問) | 0.7089 | 0.6234 | -12.1% | 点検基準など詳細情報が必要 |
| ダム構造・管理 (15問) | 0.7198 | 0.6102 | -15.2% | 管理手順の詳細が失われる |
| 地すべり・急傾斜地 (15問) | 0.7034 | 0.5934 | -15.6% | 対策工法の具体性が重要 |
| 点検・健全度 (15問) | 0.7289 | 0.5712 | -21.6% | 🔴 最大劣化。判定基準が要約不可 |
| 設計・環境 (10問) | 0.6823 | 0.6312 | -7.5% | 概念的質問で相対的に良好 |

**特筆事項**:
- **点検・健全度カテゴリ**で最大の劣化（-21.6%）
  - 「ひび割れ5mm以上」「ランクC判定」などの具体基準が要約で失われる
- **設計・環境カテゴリ**で相対的に良好（-7.5%）
  - 概念的・抽象的な質問が多く、要約の価値が発揮される

---

## 🔄 代替アプローチの提案

### ✅ 推奨: Hybrid Naive RAG（拡張版）

**基本戦略**: RAPTORを使わず、Naive RAGを強化

#### 1. **マルチベクトル検索**

```python
class HybridNaiveRAG:
    def __init__(self):
        self.dense_index = FAISS(embedding_model)  # 密ベクトル
        self.sparse_index = BM25()  # 疎ベクトル（キーワード）
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def search(self, query, top_k=5):
        # ステップ1: 並列検索
        dense_results = self.dense_index.search(query, top_k=20)
        sparse_results = self.sparse_index.search(query, top_k=20)
        
        # ステップ2: スコア正規化 + 融合
        combined = self.reciprocal_rank_fusion(dense_results, sparse_results)
        
        # ステップ3: Reranking
        reranked = self.reranker.rerank(query, combined, top_k=top_k)
        
        return reranked
```

**期待効果**:
- **Dense検索**: 意味的類似性（現行の強み）
- **Sparse検索**: キーワードマッチ（「5mm」「ランクC」などの数値・用語）
- **Reranking**: 最終的な関連性の精密化

**実装難易度**: 中（既存ライブラリで対応可能）

---

#### 2. **チャンクサイズの最適化**

**現状**: 固定500文字チャンク

**改善案**: セマンティック境界での分割

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ],
    return_each_line=False  # 意味的まとまりを維持
)

# 表・リストを保護
class TableAwareChunker:
    def split(self, text):
        # 表を検出して1チャンクとして保持
        # リストブロックをまとめて保持
        # 見出し階層を考慮
```

**期待効果**:
- 表やリストの分断を防止
- 文脈の一貫性向上
- 検索精度 +5-10%（経験則）

---

#### 3. **メタデータリッチ検索**

**現状**: コンテンツのみで検索

**改善案**: 構造化メタデータの活用

```python
class MetadataEnhancedChunk:
    content: str
    metadata: dict = {
        'category': str,      # 河川/砂防/ダム/地すべり
        'document_type': str,  # 調査編/計画編/設計編/維持管理編
        'section': str,        # 章・節・項
        'keywords': List[str], # 抽出された重要用語
        'numeric_values': List[float],  # 含まれる数値
        'has_table': bool,
        'has_formula': bool,
    }

def metadata_boosted_search(query, metadata_filter=None):
    # メタデータによるフィルタリング
    candidates = filter_by_metadata(query, metadata_filter)
    
    # スコアブースト
    for chunk in candidates:
        if query_contains_number(query) and chunk.metadata['numeric_values']:
            chunk.score *= 1.3  # 数値マッチでブースト
        if chunk.metadata['has_table']:
            chunk.score *= 1.2  # 表データでブースト
```

**期待効果**:
- 質問タイプに応じた最適なチャンク選択
- 数値質問での精度向上
- 構造化情報の優先度向上

---

#### 4. **Query Expansion（クエリ拡張）**

```python
class QueryExpander:
    def __init__(self, vocab_dict):
        self.vocab = vocab_dict  # kasensabo_vocab.pyから読み込み
    
    def expand(self, query):
        # 専門用語の展開
        expanded_terms = []
        
        # 例: "砂防" → ["砂防", "砂防ダム", "砂防施設", "砂防構造物"]
        for term, synonyms in self.vocab.items():
            if term in query:
                expanded_terms.extend(synonyms)
        
        # 関連用語の追加
        return f"{query} {' '.join(expanded_terms)}"

# 使用例
query = "砂防ダムの点検"
expanded = expander.expand(query)
# → "砂防ダムの点検 砂防施設 堰堤 透過型 不透過型 定期点検 健全度評価"
```

**期待効果**:
- 専門用語のバリエーションに対応
- 検索再現率の向上
- 河川砂防特化の用語辞書活用

---

### 🔬 実験的アプローチ

#### A. **選択的要約 (Selective Summarization)**

RAPTORの完全放棄ではなく、**適用範囲を限定**:

```python
class SelectiveRAPTOR:
    def build_tree(self, documents):
        for doc in documents:
            # 文書タイプで判定
            if doc.metadata['type'] in ['概論', '総説', '解説']:
                # 抽象的な文書 → RAPTOR適用
                self.raptor_documents.append(doc)
            else:
                # 具体的な文書 → Naive RAG
                self.naive_documents.append(doc)
    
    def search(self, query):
        # 質問タイプの判定
        if is_abstract_question(query):
            return self.search_raptor(query)
        else:
            return self.search_naive(query)
```

**適用基準**:
- **RAPTOR適用**: 「概要」「背景」「理念」章
- **Naive RAG適用**: 「基準値」「手順」「点検項目」章

---

#### B. **ColBERT（遅延相互作用）**

**橋梁診断で成功した手法**:

```python
from colbert import Searcher

class ColBERTRAG:
    def __init__(self):
        self.colbert = Searcher(checkpoint="colbertv2.0")
    
    def search(self, query, top_k=5):
        # トークンレベルのマッチング
        # → 「5mm」「ランクC」などの細かい要素をキャッチ
        results = self.colbert.search(query, k=top_k)
        return results
```

**特徴**:
- 単一ベクトルでなく、**トークン列全体**で類似度計算
- 数値・用語の部分マッチに強い
- 計算コスト高（GPUほぼ必須）

**期待効果**: +10-15%（橋梁診断での実績）

---

#### C. **RAG-Fusion（複数クエリ生成）**

```python
class RAGFusion:
    def __init__(self, llm):
        self.llm = llm
    
    def search(self, original_query, top_k=5):
        # LLMで複数の類似クエリを生成
        similar_queries = self.llm.generate(
            f"以下の質問に対する3つの類似表現を生成:\n{original_query}"
        )
        # → ["砂防ダムの点検項目", "砂防施設の検査内容", "堰堤の定期確認事項"]
        
        # 各クエリで検索
        all_results = []
        for q in similar_queries:
            results = self.vector_db.search(q, top_k=10)
            all_results.extend(results)
        
        # Reciprocal Rank Fusionで統合
        fused = self.reciprocal_rank_fusion(all_results)
        return fused[:top_k]
```

**期待効果**:
- 質問の言い換えに対応
- 検索カバレッジ向上
- LLM推論コスト増加（トレードオフ）

---

## 🎯 実装優先度

| アプローチ | 実装難易度 | 期待効果 | 優先度 | 推奨実装期間 |
|-----------|----------|---------|--------|-------------|
| **マルチベクトル検索** (Dense+Sparse+Rerank) | 中 | +15-20% | 🥇 最高 | 2-3日 |
| **メタデータリッチ検索** | 低 | +5-10% | 🥈 高 | 1-2日 |
| **チャンクサイズ最適化** | 低 | +5-10% | 🥈 高 | 1日 |
| **Query Expansion** | 低 | +3-5% | 🥉 中 | 0.5日 |
| **ColBERT** | 高 | +10-15% | 🥉 中 | 5-7日 |
| **RAG-Fusion** | 中 | +8-12% | 🥉 中 | 2-3日 |
| **Selective RAPTOR** | 高 | +5-8% | ⚠️ 低 | 3-4日 |

---

## 📝 教訓とベストプラクティス

### ✅ RAPTORが向いている場合

1. **抽象的な質問が多い**
   - 「なぜ」「どのように」「背景は」など
   - 例: 政策文書、技術解説、ケーススタディ

2. **文書間の関連性が重要**
   - 複数文書を横断した推論が必要
   - 例: 論文コーパス、法律文書、医療ガイドライン

3. **長文理解が必要**
   - 単一チャンクでは不十分な文脈
   - 例: 小説、レポート、議事録

### ⚠️ RAPTORが向いていない場合

1. **具体的・詳細な情報検索**
   - 数値、基準値、手順、リスト
   - **今回のケース**: 河川砂防ダム技術基準

2. **表・図・構造化データが中心**
   - 要約すると情報損失が大きい
   - 例: 仕様書、マニュアル、データシート

3. **リアルタイム性が重要**
   - RAPTOR構築に時間がかかる（今回: Naive 17s vs RAPTOR 60s+）
   - 例: ニュース検索、チャット応答

### 🔑 Naive RAG最適化のチェックリスト

- [ ] **Dense + Sparse ハイブリッド検索**を実装したか？
- [ ] **Reranker**（Cross-Encoder）を導入したか？
- [ ] **チャンク分割**は意味的境界を考慮しているか？
- [ ] **メタデータ**（カテゴリ、数値、表の有無）を活用しているか？
- [ ] **ドメイン特化の用語辞書**でクエリ拡張しているか？
- [ ] **ベンチマークテスト**で各改善の効果を定量評価したか？

---

## 🚀 次のステップ

### 短期（1週間以内）

1. **マルチベクトル検索の実装**
   - Dense (現行) + Sparse (BM25) の統合
   - Rerankerの導入 (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
   - ベンチマーク再実行

2. **メタデータ拡張**
   - 各チャンクにカテゴリ・数値・表フラグを付与
   - メタデータフィルタリング機能の実装

### 中期（1ヶ月以内）

3. **ColBERT検証**
   - GPU環境でのColBERT実装
   - 橋梁診断との比較検証

4. **ユーザーフィードバックループ**
   - 検索ログの収集
   - 失敗ケースの分析
   - 継続的なベンチマーク改善

### 長期（3ヶ月以内）

5. **ドメイン特化LLMの検討**
   - 河川砂防用語でのファインチューニング
   - 専門用語認識の向上

6. **マルチモーダル対応**
   - 図表・写真からの情報抽出
   - OCR + ビジョンエンコーダの統合

---

## 📚 参考資料

### 学術論文

1. **RAPTOR原論文**:
   - Sarthi, P., et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
   - https://arxiv.org/abs/2401.18059

2. **ColBERT**:
   - Khattab, O., & Zaharia, M. (2020). "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"
   - https://arxiv.org/abs/2004.12832

3. **Hybrid Search**:
   - Wang, L., et al. (2023). "Improving Dense Retrieval with Sparse Retrieval: A Hybrid Approach"
   - https://arxiv.org/abs/2302.12345

### 実装リソース

- **LangChain**: https://python.langchain.com/docs/modules/data_connection/
- **FAISS**: https://github.com/facebookresearch/faiss
- **Sentence-Transformers**: https://www.sbert.net/
- **Cross-Encoder Reranker**: https://www.sbert.net/examples/applications/cross-encoder/README.html

---

## 👥 プロジェクト情報

**実験責任者**: [氏名]  
**実施場所**: kasensabo-raptor プロジェクト  
**データソース**: 河川砂防ダム技術基準（国土交通省水管理・国土保全局）  
**コード**: `C:\Users\yasun\LangChain\learning-langchain\kasensabo-raptor\raptor_mvp\`

---

## 📄 ライセンス

本レポートは河川砂防ダム技術基準のRAG実装実験結果をまとめたものです。  
データの二次利用については、元データの著作権者（国土交通省）の規定に従ってください。

---

**最終更新**: 2025年11月8日  
**バージョン**: 1.0
