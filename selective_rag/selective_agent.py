"""
Selective Agent - 質問の専門性粒度を判定してRAGシステムを選択
"""

import logging
from typing import Literal, Dict
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import Ollama
from config import SELECTOR_MODEL, SELECTOR_API_BASE, SELECTOR_USE_LOCAL

logger = logging.getLogger(__name__)


class SelectiveAgent:
    """
    質問の専門性粒度を判定し、最適なRAGシステムを選択するエージェント
    
    - Fine-grained (細かい粒度): 具体的・数値的 → ColBERT RAG
    - Coarse-grained (粗い粒度): 概念的・抽象的 → Naive RAG
    """
    
    def __init__(self):
        """LLMベースのセレクター初期化"""
        try:
            if SELECTOR_USE_LOCAL:
                # ローカルモデルを使用
                logger.info(f"Loading local model from {SELECTOR_MODEL}")
                self.tokenizer = AutoTokenizer.from_pretrained(SELECTOR_MODEL)
                self.model = AutoModelForCausalLM.from_pretrained(
                    SELECTOR_MODEL,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                self.llm = None
                logger.info(f"✓ Selector Agent initialized with local model")
            else:
                # Ollamaを使用
                self.llm = Ollama(
                    model=SELECTOR_MODEL,
                    base_url=SELECTOR_API_BASE,
                    temperature=0.1,
                )
                self.tokenizer = None
                self.model = None
                logger.info(f"✓ Selector Agent initialized with Ollama")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")
            logger.info("Falling back to rule-based selector")
            self.llm = None
            self.tokenizer = None
            self.model = None
    
    def classify_question(self, question: str) -> Literal["fine", "coarse"]:
        """
        質問の専門性粒度を判定
        
        Args:
            question: 評価する質問文
        
        Returns:
            "fine": 細かい粒度（ColBERT推奨）
            "coarse": 粗い粒度（Naive推奨）
        """
        if self.llm or self.model:
            return self._llm_classify(question)
        else:
            return self._rule_based_classify(question)
    
    def _llm_classify(self, question: str) -> Literal["fine", "coarse"]:
        """LLMベースの分類"""
        prompt = f"""以下の質問を分析し、専門性の粒度を判定してください。

【判定基準】
- Fine（細かい粒度）: 具体的な数値、材料規格、計算式、測定方法、寸法など詳細情報を求める質問
- Coarse（粗い粒度）: 概念、定義、目的、比較、全体像、原則など抽象的・概念的な理解を求める質問

【質問】
{question}

【指示】
この質問が「Fine」か「Coarse」のどちらに該当するか、一言で答えてください。
理由は不要です。「Fine」または「Coarse」とだけ回答してください。

回答:"""

        try:
            if self.llm:
                # Ollama使用
                response = self.llm.invoke(prompt)
            else:
                # ローカルモデル使用
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            response_lower = response.strip().lower()
            
            if "fine" in response_lower:
                return "fine"
            elif "coarse" in response_lower:
                return "coarse"
            else:
                # デフォルトはルールベースにフォールバック
                logger.warning(f"Unexpected LLM response: {response}. Falling back to rule-based.")
                return self._rule_based_classify(question)
                
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return self._rule_based_classify(question)
    
    def _rule_based_classify(self, question: str) -> Literal["fine", "coarse"]:
        """ルールベースの分類（フォールバック）"""
        # 細かい粒度のキーワード
        fine_keywords = [
            # 数値関連
            "何メートル", "何センチ", "何ミリ", "何キロ", "何トン", "何kg", "何g",
            "何年", "何月", "何日", "何時間", "何分", "何秒",
            "何度", "何℃", "何%", "何倍", "何割", "何分の",
            "いくつ", "最低", "最大", "最小", "上限", "下限",
            
            # 具体的情報
            "規格", "仕様", "材質", "寸法", "サイズ", "直径", "幅", "高さ", "長さ", "厚さ",
            "計算式", "係数", "記号", "単位", "式", "算定",
            "測定方法", "試験方法", "手順", "頻度", "間隔",
            "JIS", "番号", "型番", "品番",
            "配合", "成分", "含有",
        ]
        
        # 粗い粒度のキーワード
        coarse_keywords = [
            # 概念・定義
            "とは", "について説明", "概念", "定義", "意味",
            
            # 目的・理由
            "目的", "理由", "なぜ", "背景", "経緯",
            
            # 比較・関係
            "違い", "比較", "関係", "関連", "関係性", "対比",
            
            # 全体像
            "全体", "体系", "構造", "仕組み", "プロセス", "フロー", "流れ",
            "段階", "手続き",
            
            # 原則・考え方
            "原則", "考え方", "基本", "理念", "方針",
        ]
        
        question_lower = question.lower()
        
        # スコアリング
        fine_score = sum(1 for kw in fine_keywords if kw in question_lower)
        coarse_score = sum(1 for kw in coarse_keywords if kw in question_lower)
        
        # 判定
        if fine_score > coarse_score:
            return "fine"
        elif coarse_score > fine_score:
            return "coarse"
        else:
            # 同点の場合は質問の長さで判定（短い→具体的、長い→抽象的）
            if len(question) < 30:
                return "fine"
            else:
                return "coarse"
    
    def select_rag_system(self, question: str) -> Dict[str, str]:
        """
        質問に対して最適なRAGシステムを選択
        
        Args:
            question: 質問文
        
        Returns:
            選択情報の辞書
        """
        granularity = self.classify_question(question)
        
        if granularity == "fine":
            system = "colbert"
            reason = "具体的・詳細な情報を求める質問のため、トークンレベル検索が有効"
        else:
            system = "naive"
            reason = "概念的・抽象的な理解を求める質問のため、高速な文レベル検索で十分"
        
        result = {
            "question": question,
            "granularity": granularity,
            "selected_system": system,
            "reason": reason,
        }
        
        logger.debug(f"Question: '{question[:50]}...' -> {system} ({granularity})")
        
        return result


def test_selector():
    """セレクターのテスト"""
    agent = SelectiveAgent()
    
    test_questions = [
        "堤防の天端幅の標準値は何メートルですか？",  # Fine
        "コンクリートの設計基準強度は何N/mm²ですか？",  # Fine
        "マニングの式における粗度係数の記号は何ですか？",  # Fine
        "河川管理とは何ですか？",  # Coarse
        "砂防ダムの役割について説明してください",  # Coarse
        "堤防と護岸の違いは何ですか？",  # Coarse
    ]
    
    print("\n=== Selective Agent Test ===")
    for q in test_questions:
        result = agent.select_rag_system(q)
        print(f"\nQ: {q}")
        print(f"  → Granularity: {result['granularity']}")
        print(f"  → System: {result['selected_system']}")
        print(f"  → Reason: {result['reason']}")


if __name__ == "__main__":
    test_selector()
