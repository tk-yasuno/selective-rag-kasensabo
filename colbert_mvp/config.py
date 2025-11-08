"""
ColBERT MVP Configuration
河川砂防ダムColBERT検索システム設定
"""
from pathlib import Path

# データパス
DATA_DIR = Path(r"C:\Users\yasun\LangChain\learning-langchain\kasensabo-raptor\data\kasensabo_knowledge_base")
OUTPUT_DIR = Path(r"C:\Users\yasun\LangChain\learning-langchain\kasensabo-raptor\colbert_mvp\output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ColBERT設定
COLBERT_MODEL = "colbert-ir/colbertv2.0"  # HuggingFace標準モデル
COLBERT_INDEX_NAME = "kasensabo_colbert_index"
COLBERT_CHECKPOINT = OUTPUT_DIR / "colbert_checkpoint"

# 文書処理設定
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MAX_DOCUMENT_LENGTH = 512  # ColBERTのトークン制限
SAMPLE_RATIO = 1.0  # 文書サンプリング比率（1.0 = 100%全文書）

# 検索設定
TOP_K_RETRIEVAL = 5
NCELLS = 2  # ColBERTのインデックスパラメータ
CENTROID_SCORE_THRESHOLD = 0.5
NDOCS = 256  # ColBERT検索の候補数

# ベンチマーク設定
RAPTOR_OUTPUT_DIR = Path(r"C:\Users\yasun\LangChain\learning-langchain\kasensabo-raptor\raptor_mvp\output")
BENCHMARK_QUESTIONS_FILE = RAPTOR_OUTPUT_DIR / "benchmark_questions_100.json"
BENCHMARK_RESULTS_FILE = OUTPUT_DIR / f"colbert_benchmark_results_{int(SAMPLE_RATIO*100)}pct.json"

# デバイス設定
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_AVAILABLE = torch.cuda.is_available()

if GPU_AVAILABLE:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🚀 GPU: {GPU_NAME} ({GPU_MEMORY:.1f}GB)")
else:
    print("⚠️ CPU mode (ColBERT推奨: GPU)")

# ログ設定
LOG_LEVEL = "INFO"
