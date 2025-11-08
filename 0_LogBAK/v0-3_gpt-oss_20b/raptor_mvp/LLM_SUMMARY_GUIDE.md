# 河川砂防RAPTOR - 要約LLMの現状と推奨

## 📊 現在の実装状況

### 現在のMVP（raptor_mvp/raptor_rag.py）
**要約方法**: **LLMを使用していない（単純な文字列結合）**

```python
def _create_summary(self, node_ids: List[str]) -> str:
    """クラスタから要約を生成（簡易版：先頭部分を結合）"""
    texts = [self.nodes[nid].content for nid in node_ids[:5]]  # 最大5件
    combined = "\n\n".join([t[:200] for t in texts])
    
    if len(combined) > 800:
        combined = combined[:800] + "..."
    
    return combined
```

**問題点**:
- ❌ 実際の要約が生成されない（単なる文字列切り取り）
- ❌ 冗長な情報が含まれる
- ❌ 文脈の理解がない

---

## 🚀 ベースコード（true_raptor_builder.py）の実装

### 使用しているLLM（16GB GPU対応）

```python
# GPU容量に応じた自動選択
if gpu_memory >= 24:  # 24GB以上
    llm_model_name = "facebook/opt-6.7b"  # 6.7Bパラメータ
elif gpu_memory >= 16:  # 16GB以上 ← あなたのケース
    llm_model_name = "facebook/opt-2.7b"  # 2.7Bパラメータ
elif gpu_memory >= 12:
    llm_model_name = "facebook/opt-1.3b"
elif gpu_memory >= 8:
    llm_model_name = "microsoft/DialoGPT-large"
else:
    llm_model_name = "microsoft/DialoGPT-medium"
```

**16GB GPUの場合**: `facebook/opt-2.7b` を使用

### 要約生成の実装

```python
def generate_llm_summary(self, documents: List[str]) -> str:
    """GPU対応の大規模LLMを使用してクラスタの要約を生成"""
    
    # プロンプト作成（河川砂防用にカスタマイズ可能）
    prompt = f"""Summarize the following findings in a concise scientific manner.
Focus on key mechanisms and processes.

Findings: {combined_text}

Summary:"""
    
    # トークン化
    inputs = self.llm_tokenizer.encode(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=800,
        padding=True
    )
    
    # 生成（GPU最適化）
    with torch.no_grad():
        outputs = self.llm_model.generate(
            inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    summary = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary[:400]
```

---

## 🎯 河川砂防ダム技術基準に最適なLLM選択肢

### 1️⃣ **推奨: 日本語特化モデル（16GB GPU対応）**

#### **elyza/ELYZA-japanese-Llama-2-7b** ⭐ 最推奨
- **パラメータ**: 7B
- **特徴**: 日本語に特化したLlama 2
- **メモリ**: FP16で約14GB（16GBに収まる）
- **利点**: 
  - ✅ 日本語の技術文書に最適
  - ✅ 河川砂防の専門用語を正しく理解
  - ✅ 16GBで快適に動作

```python
llm_model_name = "elyza/ELYZA-japanese-Llama-2-7b"
self.llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
```

#### **rinna/japanese-gpt-neox-3.6b**
- **パラメータ**: 3.6B
- **特徴**: 日本語GPT-NeoX
- **メモリ**: FP16で約7GB
- **利点**:
  - ✅ より軽量で高速
  - ✅ メモリに余裕

#### **cyberagent/open-calm-7b**
- **パラメータ**: 7B
- **特徴**: 日本語特化の大規模モデル
- **メモリ**: FP16で約14GB

---

### 2️⃣ **多言語対応モデル（英語文書も含む場合）**

#### **meta-llama/Llama-2-7b-chat-hf** ⭐ バランス型
- **パラメータ**: 7B
- **特徴**: Meta公式のチャットモデル
- **メモリ**: FP16で約14GB
- **利点**:
  - ✅ 高品質な要約生成
  - ✅ 指示追従性が高い
  - ✅ 英語・日本語両対応（質はやや低下）

#### **mistralai/Mistral-7B-Instruct-v0.2** ⭐ 高性能
- **パラメータ**: 7B
- **特徴**: 最新の効率的アーキテクチャ
- **メモリ**: FP16で約14GB
- **利点**:
  - ✅ 同サイズで最高クラスの性能
  - ✅ 長文コンテキストに強い

---

### 3️⃣ **軽量・高速モデル（余裕を持たせたい場合）**

#### **stabilityai/japanese-stablelm-instruct-alpha-7b**
- **パラメータ**: 7B
- **特徴**: 日本語StableLM
- **メモリ**: FP16で約14GB

#### **facebook/opt-2.7b**（現在のベースコード）
- **パラメータ**: 2.7B
- **特徴**: Meta OPTシリーズ
- **メモリ**: FP16で約5.4GB
- **利点**:
  - ✅ 非常に軽量
  - ❌ 日本語対応が弱い

---

## 🔧 実装例：河川砂防用にカスタマイズ

### 日本語モデルを使った要約生成

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class KasensaboRAPTORWithLLM:
    def __init__(self):
        # 日本語特化モデルを初期化
        model_name = "elyza/ELYZA-japanese-Llama-2-7b"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # メモリ効率化
            device_map="auto",          # GPU自動配置
            low_cpu_mem_usage=True
        )
        self.llm_model.eval()
    
    def _create_summary(self, node_ids: List[str]) -> str:
        """LLMを使って要約生成"""
        texts = [self.nodes[nid].content for nid in node_ids[:5]]
        combined = "\n\n".join([t[:300] for t in texts])
        
        # 河川砂防専用プロンプト
        prompt = f"""以下の河川砂防ダム技術基準の文書を、専門的かつ簡潔に要約してください。
重要な技術用語、基準値、設計手法を保持してください。

文書:
{combined}

要約:"""
        
        # トークン化
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to("cuda")
        
        # 生成
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.5,  # 技術文書なので低めに設定
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # プロンプト部分を除去
        if "要約:" in summary:
            summary = summary.split("要約:")[-1].strip()
        
        return summary[:500]
```

---

## 📊 モデル比較表（16GB GPU）

| モデル | サイズ | メモリ(FP16) | 日本語 | 速度 | 品質 | 推奨度 |
|--------|--------|--------------|--------|------|------|--------|
| **ELYZA-japanese-Llama-2-7b** | 7B | 14GB | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | **🥇** |
| **Mistral-7B-Instruct** | 7B | 14GB | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | **🥈** |
| **rinna/japanese-gpt-neox-3.6b** | 3.6B | 7GB | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | **🥉** |
| facebook/opt-2.7b (現状) | 2.7B | 5.4GB | ⭐ | ⭐⭐⭐ | ⭐ | - |

---

## ⚡ メモリ最適化テクニック

### 4bit量子化（さらに大きなモデルを使う）

```python
from transformers import BitsAndBytesConfig

# 4bit量子化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 13Bモデルも16GBで動作可能
model = AutoModelForCausalLM.from_pretrained(
    "elyza/ELYZA-japanese-Llama-2-13b",  # 13Bパラメータ
    quantization_config=bnb_config,
    device_map="auto"
)
```

**結果**: 13Bモデルが約10GBで動作

---

## 🎯 最終推奨

### **河川砂防ダム技術基準に最適**: `elyza/ELYZA-japanese-Llama-2-7b`

**理由**:
1. ✅ 日本語技術文書の理解が優秀
2. ✅ 16GBで快適に動作
3. ✅ 専門用語を正しく保持
4. ✅ 要約の品質が高い

### 実装手順

1. **モデルダウンロード**
```bash
pip install transformers accelerate bitsandbytes
```

2. **raptor_rag.pyに統合**
```python
# config.pyに追加
LLM_MODEL = "elyza/ELYZA-japanese-Llama-2-7b"
USE_LLM_SUMMARY = True

# raptor_rag.pyで初期化
if USE_LLM_SUMMARY:
    self._init_llm()
```

3. **テスト実行**
```bash
python main.py build
```

これで高品質な日本語要約が生成されます！
