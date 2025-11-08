"""
Selective RAG Configuration
専門性粒度に応じてNaive RAGとColBERT RAGを自動選択するシステムの設定
"""

from pathlib import Path
import torch

# ===== Paths =====
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR.parent / "data" / "kasensabo_knowledge_base"
OUTPUT_DIR = ROOT_DIR / "output"
LOCAL_MODEL_DIR = ROOT_DIR.parent / "models" / "ELYZA-japanese-Llama-2-7b"

# 出力ディレクトリ作成
OUTPUT_DIR.mkdir(exist_ok=True)

# ===== Document Processing =====
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_DOCUMENT_LENGTH = 512

# ===== Models =====
# Naive RAG
NAIVE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ColBERT RAG
COLBERT_MODEL = "colbert-ir/colbertv2.0"
COLBERT_SAMPLE_RATIO = 0.5  # 50%サンプリング（速度重視）

# Selector Agent (LLM)
# ローカルモデルパスを使用
SELECTOR_MODEL = str(LOCAL_MODEL_DIR)
SELECTOR_USE_LOCAL = True  # ローカルモデル使用フラグ
SELECTOR_API_BASE = "http://localhost:11434"  # Ollama使用時

# ===== Retrieval Settings =====
TOP_K_RETRIEVAL = 5

# ===== Question Dataset =====
FINE_GRAINED_QUESTIONS = 100  # 細かい粒度（具体的・数値的）
COARSE_GRAINED_QUESTIONS = 100  # 粗い粒度（概念的・抽象的）
TOTAL_QUESTIONS = FINE_GRAINED_QUESTIONS + COARSE_GRAINED_QUESTIONS

# ===== Question Types Definition =====
FINE_GRAINED_CATEGORIES = [
    "具体的な数値・基準値",
    "材料規格・仕様",
    "計算式・係数",
    "測定方法・試験手順",
    "寸法・サイズ規定",
]

COARSE_GRAINED_CATEGORIES = [
    "概念・定義の説明",
    "目的・背景の理解",
    "比較・関係性の説明",
    "全体構造・体系の理解",
    "原則・考え方の説明",
]

# ===== GPU Settings =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = True if DEVICE == "cuda" else False

# ===== Benchmark Settings =====
BENCHMARK_OUTPUT_FILE = OUTPUT_DIR / "selective_rag_benchmark_results.json"
QUESTIONS_FILE = OUTPUT_DIR / "questions_200.json"

# ===== Logging =====
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# GPU情報表示
if torch.cuda.is_available():
    print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Running on CPU")
