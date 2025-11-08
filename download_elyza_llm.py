"""
ELYZA-japanese-Llama-2-7b モデルの高速ダウンロードスクリプト
hf_transferを使用して高速にモデルをダウンロード
"""
import os
from huggingface_hub import snapshot_download

# hf_transferによる高速ダウンロードを有効化
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def download_elyza_model():
    """ELYZA-japanese-Llama-2-7bをダウンロード（safetensors版）"""
    model_name = "elyza/ELYZA-japanese-Llama-2-7b"
    
    print(f"📥 Downloading {model_name} (safetensors format)...")
    print("⚡ Using hf_transfer for accelerated download")
    print("=" * 60)
    
    try:
        local_dir = snapshot_download(
            repo_id=model_name,
            local_dir=f"./models/{model_name.split('/')[-1]}-safetensors",
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.bin", "*.pth", "pytorch_model*"]  # pytorchファイルを除外
        )
        
        print("=" * 60)
        print(f"✅ Model downloaded successfully!")
        print(f"📁 Location: {local_dir}")
        print("\n🔧 Usage in code:")
        print(f'    model = AutoModelForCausalLM.from_pretrained("{local_dir}")')
        
        return local_dir
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

if __name__ == "__main__":
    download_elyza_model()
