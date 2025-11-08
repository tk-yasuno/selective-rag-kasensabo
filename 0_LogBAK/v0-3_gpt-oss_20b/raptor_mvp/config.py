"""
Configuration for Kasensabo RAPTOR MVP
河川砂防RAPTOR MVP設定
"""
from pathlib import Path

# データパス
DATA_DIR = Path(r"C:\Users\yasun\LangChain\learning-langchain\kasensabo-raptor\data\kasensabo_knowledge_base")
VOCAB_FILE = Path(r"C:\Users\yasun\LangChain\learning-langchain\kasensabo-raptor\kasensabo_vocab.py")
OUTPUT_DIR = Path(r"C:\Users\yasun\LangChain\learning-langchain\kasensabo-raptor\raptor_mvp\output")

# モデル設定
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 高速・軽量
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # 高精度

# RAG設定
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 5

# RAPTOR設定
MAX_TREE_DEPTH = 3
MIN_CLUSTER_SIZE = 3
MAX_CLUSTER_SIZE = 30
MIN_CLUSTERS = 2
MAX_CLUSTERS = 5

# クラスタリング評価戦略
CLUSTERING_STRATEGY = "balanced"  # "silhouette", "dbi", "balanced"
METRIC_WEIGHTS = {
    'silhouette': 0.5,
    'dbi': 0.5
}

# LLM要約設定
USE_LLM_SUMMARY = True  # Trueでリアル要約、Falseで簡易結合
USE_OLLAMA = True  # TrueでOllama、FalseでHuggingFace transformers
OLLAMA_MODEL = "gpt-oss:20b"  # Ollamaモデル名（橋梁診断で成功した高品質モデル）
LLM_MODEL = r"C:\Users\yasun\LangChain\learning-langchain\kasensabo-raptor\models\ELYZA-japanese-Llama-2-7b-safetensors"  # transformers用
LLM_MAX_NEW_TOKENS = 150  # 要約の最大トークン数
LLM_TEMPERATURE = 0.5  # 技術文書なので低めに設定
LLM_TOP_P = 0.9
LLM_REPETITION_PENALTY = 1.2

# ベンチマーク設定
BENCHMARK_QUESTIONS_FILE = OUTPUT_DIR / "benchmark_questions_100.json"
BENCHMARK_RESULTS_FILE = OUTPUT_DIR / "benchmark_results.json"

# デバイス設定
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ログ設定
LOG_LEVEL = "INFO"
