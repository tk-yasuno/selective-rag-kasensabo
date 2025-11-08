"""
Question Generator for Selective RAG
細かい粒度100問と粗い粒度100問を生成
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
from config import (
    QUESTIONS_FILE, 
    FINE_GRAINED_QUESTIONS, 
    COARSE_GRAINED_QUESTIONS,
    FINE_GRAINED_CATEGORIES,
    COARSE_GRAINED_CATEGORIES
)

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """質問データセット生成器"""
    
    def __init__(self):
        self.questions = []
    
    def generate_fine_grained_questions(self) -> List[Dict]:
        """
        細かい粒度の質問を生成（具体的・数値的・詳細）
        ColBERTが得意とする質問タイプ
        """
        fine_questions = []
        
        # 具体的な数値・基準値（20問）
        fine_questions.extend([
            {"id": 1, "category": "具体的な数値・基準値", "granularity": "fine", 
             "question": "堤防の天端幅の標準値は何メートルですか？"},
            {"id": 2, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "基本高水の設計確率年は何年ですか？"},
            {"id": 3, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "コンクリートの設計基準強度は何N/mm²ですか？"},
            {"id": 4, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "ダムの安全率は最小いくつ必要ですか？"},
            {"id": 5, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "砂防ダムの高さの上限は何メートルですか？"},
            {"id": 6, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "河川の計画高水流量の算定確率は何年確率ですか？"},
            {"id": 7, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "護岸の根入れ深さは最低何メートル必要ですか？"},
            {"id": 8, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "土石流の流速は最大で時速何キロメートルですか？"},
            {"id": 9, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "ダムの堆砂容量は計画の何％確保する必要がありますか？"},
            {"id": 10, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "洪水時の余裕高は波高の何倍必要ですか？"},
            {"id": 11, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "鋼材の許容応力度は降伏点の何％ですか？"},
            {"id": 12, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "地震時の震度階級は最大いくつまでですか？"},
            {"id": 13, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "水位計の測定精度は何センチメートルですか？"},
            {"id": 14, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "透水係数の標準値は何cm/secですか？"},
            {"id": 15, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "点検の実施頻度は年に何回ですか？"},
            {"id": 16, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "護岸ブロックの重量は何トン以上ですか？"},
            {"id": 17, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "河床勾配の最大値は何パーセントですか？"},
            {"id": 18, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "ダムの越流部の幅は何メートル必要ですか？"},
            {"id": 19, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "粗度係数nの標準値はいくつですか？"},
            {"id": 20, "category": "具体的な数値・基準値", "granularity": "fine",
             "question": "設計地震動のマグニチュードは最低いくつですか？"},
        ])
        
        # 材料規格・仕様（20問）
        fine_questions.extend([
            {"id": 21, "category": "材料規格・仕様", "granularity": "fine",
             "question": "コンクリートの配合で使用するセメントの種類は何ですか？"},
            {"id": 22, "category": "材料規格・仕様", "granularity": "fine",
             "question": "鋼材のJIS規格番号は何ですか？"},
            {"id": 23, "category": "材料規格・仕様", "granularity": "fine",
             "question": "骨材の最大寸法は何ミリメートルですか？"},
            {"id": 24, "category": "材料規格・仕様", "granularity": "fine",
             "question": "防水シートの材質は何を使用しますか？"},
            {"id": 25, "category": "材料規格・仕様", "granularity": "fine",
             "question": "護岸用石材の比重は最低いくつ必要ですか？"},
            {"id": 26, "category": "材料規格・仕様", "granularity": "fine",
             "question": "アンカーボルトの材質規格は何ですか？"},
            {"id": 27, "category": "材料規格・仕様", "granularity": "fine",
             "question": "ジオテキスタイルの引張強度は何kN/mですか？"},
            {"id": 28, "category": "材料規格・仕様", "granularity": "fine",
             "question": "鉄筋の直径規格にはどのサイズがありますか？"},
            {"id": 29, "category": "材料規格・仕様", "granularity": "fine",
             "question": "吸出し防止材の透水係数は何cm/sec以下ですか？"},
            {"id": 30, "category": "材料規格・仕様", "granularity": "fine",
             "question": "コンクリートの水セメント比は何％以下ですか？"},
            {"id": 31, "category": "材料規格・仕様", "granularity": "fine",
             "question": "地盤改良材の種類にはどのようなものがありますか？"},
            {"id": 32, "category": "材料規格・仕様", "granularity": "fine",
             "question": "防錆塗装の膜厚は何ミクロン必要ですか？"},
            {"id": 33, "category": "材料規格・仕様", "granularity": "fine",
             "question": "止水材の許容変形量は何％ですか？"},
            {"id": 34, "category": "材料規格・仕様", "granularity": "fine",
             "question": "路盤材の粒度分布の規格値は何ですか？"},
            {"id": 35, "category": "材料規格・仕様", "granularity": "fine",
             "question": "充填材の密度は何g/cm³以上必要ですか？"},
            {"id": 36, "category": "材料規格・仕様", "granularity": "fine",
             "question": "継手用シール材の耐久年数は何年ですか？"},
            {"id": 37, "category": "材料規格・仕様", "granularity": "fine",
             "question": "遮水シートの厚さは最低何ミリメートルですか？"},
            {"id": 38, "category": "材料規格・仕様", "granularity": "fine",
             "question": "補強繊維の長さは何センチメートルですか？"},
            {"id": 39, "category": "材料規格・仕様", "granularity": "fine",
             "question": "表面処理剤の塗布量は何kg/m²ですか？"},
            {"id": 40, "category": "材料規格・仕様", "granularity": "fine",
             "question": "プレキャスト製品のJIS規格番号は何ですか？"},
        ])
        
        # 計算式・係数（20問）
        fine_questions.extend([
            {"id": 41, "category": "計算式・係数", "granularity": "fine",
             "question": "マニングの式における粗度係数の記号は何ですか？"},
            {"id": 42, "category": "計算式・係数", "granularity": "fine",
             "question": "流量計算式Q=A×Vの各記号の意味は何ですか？"},
            {"id": 43, "category": "計算式・係数", "granularity": "fine",
             "question": "安全率の計算式はどのように表されますか？"},
            {"id": 44, "category": "計算式・係数", "granularity": "fine",
             "question": "クーロンの土圧係数Kaの計算式は何ですか？"},
            {"id": 45, "category": "計算式・係数", "granularity": "fine",
             "question": "ダルシー・ワイスバッハの式の摩擦損失係数λはいくつですか？"},
            {"id": 46, "category": "計算式・係数", "granularity": "fine",
             "question": "せん断抵抗角φの標準値は何度ですか？"},
            {"id": 47, "category": "計算式・係数", "granularity": "fine",
             "question": "粘着力cの単位は何ですか？"},
            {"id": 48, "category": "計算式・係数", "granularity": "fine",
             "question": "水平震度khの算定式はどのようなものですか？"},
            {"id": 49, "category": "計算式・係数", "granularity": "fine",
             "question": "弾性係数Eの標準値は何N/mm²ですか？"},
            {"id": 50, "category": "計算式・係数", "granularity": "fine",
             "question": "ポアソン比νの一般的な値はいくつですか？"},
            {"id": 51, "category": "計算式・係数", "granularity": "fine",
             "question": "流速係数の算定に用いる径深Rの定義は何ですか？"},
            {"id": 52, "category": "計算式・係数", "granularity": "fine",
             "question": "土の単位体積重量γの標準値は何kN/m³ですか？"},
            {"id": 53, "category": "計算式・係数", "granularity": "fine",
             "question": "支持力係数Nγの値はどのように求めますか？"},
            {"id": 54, "category": "計算式・係数", "granularity": "fine",
             "question": "動水勾配iの計算式は何ですか？"},
            {"id": 55, "category": "計算式・係数", "granularity": "fine",
             "question": "揚圧力の計算に用いる係数αの値は何ですか？"},
            {"id": 56, "category": "計算式・係数", "granularity": "fine",
             "question": "遮水壁の厚さ算定式における係数は何ですか？"},
            {"id": 57, "category": "計算式・係数", "granularity": "fine",
             "question": "曲げモーメントMの算定式はどのようなものですか？"},
            {"id": 58, "category": "計算式・係数", "granularity": "fine",
             "question": "反力係数の単位は何ですか？"},
            {"id": 59, "category": "計算式・係数", "granularity": "fine",
             "question": "沈下量の推定式における圧密係数Cvの単位は何ですか？"},
            {"id": 60, "category": "計算式・係数", "granularity": "fine",
             "question": "浸透流量の計算式におけるkの記号は何を表しますか？"},
        ])
        
        # 測定方法・試験手順（20問）
        fine_questions.extend([
            {"id": 61, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "現場透水試験の手順を教えてください"},
            {"id": 62, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "コア採取の標準的な深度間隔は何メートルですか？"},
            {"id": 63, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "圧縮強度試験の供試体寸法は何センチメートルですか？"},
            {"id": 64, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "平板載荷試験の載荷板の直径は何センチメートルですか？"},
            {"id": 65, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "標準貫入試験のN値の測定方法は何ですか？"},
            {"id": 66, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "水位観測の測定頻度は最低何時間おきですか？"},
            {"id": 67, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "濁度測定の単位は何ですか？"},
            {"id": 68, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "粒度試験のふるいの目開きサイズは何種類使用しますか？"},
            {"id": 69, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "地下水位の測定方法にはどのようなものがありますか？"},
            {"id": 70, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "ひび割れ幅の測定に使用する機器は何ですか？"},
            {"id": 71, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "鉄筋の間隔測定精度は何ミリメートルですか？"},
            {"id": 72, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "コンクリートの空隙率測定方法は何ですか？"},
            {"id": 73, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "音響測定による劣化診断の周波数範囲は何Hzですか？"},
            {"id": 74, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "変位計の測定レンジは何ミリメートルですか？"},
            {"id": 75, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "傾斜計の測定精度は何度ですか？"},
            {"id": 76, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "流速計の測定位置は水深の何割の位置ですか？"},
            {"id": 77, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "土質試験の含水比測定は何℃で乾燥させますか？"},
            {"id": 78, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "pH測定の校正液の種類は何を使用しますか？"},
            {"id": 79, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "浮遊物質量の測定フィルターの孔径は何μmですか？"},
            {"id": 80, "category": "測定方法・試験手順", "granularity": "fine",
             "question": "超音波探傷試験の周波数は何MHzですか？"},
        ])
        
        # 寸法・サイズ規定（20問）
        fine_questions.extend([
            {"id": 81, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "ダムの天端の最小幅は何メートルですか？"},
            {"id": 82, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "護岸ブロックの標準寸法は何センチメートルですか？"},
            {"id": 83, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "堤防の法面勾配は何対何ですか？"},
            {"id": 84, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "放水路の最小断面積は何m²ですか？"},
            {"id": 85, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "排水管の最小口径は何ミリメートルですか？"},
            {"id": 86, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "点検用通路の幅員は最低何メートル必要ですか？"},
            {"id": 87, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "管理用道路の幅員標準は何メートルですか？"},
            {"id": 88, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "魚道の幅は最低何メートルですか？"},
            {"id": 89, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "階段の蹴上げ高さは何センチメートル以下ですか？"},
            {"id": 90, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "手すりの高さは地上何センチメートルですか？"},
            {"id": 91, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "ゲートの開閉速度は毎分何メートルですか？"},
            {"id": 92, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "越流堤の高さは何メートル以下ですか？"},
            {"id": 93, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "水叩きの長さは落差の何倍ですか？"},
            {"id": 94, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "基礎地盤の掘削深さは最低何メートルですか？"},
            {"id": 95, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "伸縮継手の間隔は何メートルごとですか？"},
            {"id": 96, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "コンクリート打継目の間隔は何メートル以内ですか？"},
            {"id": 97, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "鉄筋のかぶり厚は最低何センチメートルですか？"},
            {"id": 98, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "目地幅は何ミリメートル確保しますか？"},
            {"id": 99, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "床版の最小厚さは何センチメートルですか？"},
            {"id": 100, "category": "寸法・サイズ規定", "granularity": "fine",
             "question": "アンカーの定着長さは直径の何倍ですか？"},
        ])
        
        return fine_questions
    
    def generate_coarse_grained_questions(self) -> List[Dict]:
        """
        粗い粒度の質問を生成（概念的・抽象的・全体像）
        Naive RAGが対応可能な質問タイプ
        """
        coarse_questions = []
        
        # 概念・定義の説明（20問）
        coarse_questions.extend([
            {"id": 101, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "河川管理とは何ですか？"},
            {"id": 102, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "砂防ダムの役割について説明してください"},
            {"id": 103, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "洪水調節の概念を教えてください"},
            {"id": 104, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "流域治水とはどのような考え方ですか？"},
            {"id": 105, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "河川環境の保全とは何を指しますか？"},
            {"id": 106, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "土石流災害とはどのような現象ですか？"},
            {"id": 107, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "堤防の機能について説明してください"},
            {"id": 108, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "河川の連続性とは何ですか？"},
            {"id": 109, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "多自然川づくりの理念を教えてください"},
            {"id": 110, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "維持管理の目的は何ですか？"},
            {"id": 111, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "河川整備計画とは何ですか？"},
            {"id": 112, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "水系一貫管理の意味を説明してください"},
            {"id": 113, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "減災対策とは何ですか？"},
            {"id": 114, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "治水安全度とはどのような指標ですか？"},
            {"id": 115, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "親水空間の概念を教えてください"},
            {"id": 116, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "河川区域とは何を指しますか？"},
            {"id": 117, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "水害リスクとはどのようなものですか？"},
            {"id": 118, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "河川生態系の保全について説明してください"},
            {"id": 119, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "順応的管理とは何ですか？"},
            {"id": 120, "category": "概念・定義の説明", "granularity": "coarse",
             "question": "流域マネジメントの考え方を教えてください"},
        ])
        
        # 目的・背景の理解（20問）
        coarse_questions.extend([
            {"id": 121, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "なぜ河川法が制定されたのですか？"},
            {"id": 122, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "ダム建設の目的は何ですか？"},
            {"id": 123, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "護岸工事を行う理由を説明してください"},
            {"id": 124, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "河川改修の必要性について教えてください"},
            {"id": 125, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "点検・診断を実施する目的は何ですか？"},
            {"id": 126, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "なぜ水文観測が重要なのですか？"},
            {"id": 127, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "地質調査を実施する目的を説明してください"},
            {"id": 128, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "魚道設置の背景について教えてください"},
            {"id": 129, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "河川環境整備の目的は何ですか？"},
            {"id": 130, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "なぜ流域治水が重要視されるようになったのですか？"},
            {"id": 131, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "住民参加が求められる理由を説明してください"},
            {"id": 132, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "長寿命化計画の背景は何ですか？"},
            {"id": 133, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "なぜ総合的な土砂管理が必要なのですか？"},
            {"id": 134, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "気候変動適応策を講じる理由を教えてください"},
            {"id": 135, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "河川情報の公開が重要な理由は何ですか？"},
            {"id": 136, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "なぜ段階的な整備が行われるのですか？"},
            {"id": 137, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "モニタリングを継続する目的を説明してください"},
            {"id": 138, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "予防保全の考え方が広まった背景は何ですか？"},
            {"id": 139, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "なぜ複数の対策工法を組み合わせるのですか？"},
            {"id": 140, "category": "目的・背景の理解", "granularity": "coarse",
             "question": "技術基準が改訂される理由を教えてください"},
        ])
        
        # 比較・関係性の説明（20問）
        coarse_questions.extend([
            {"id": 141, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "堤防と護岸の違いは何ですか？"},
            {"id": 142, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "河川と用水路の関係を説明してください"},
            {"id": 143, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "洪水と渇水の対策の違いについて教えてください"},
            {"id": 144, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "ダムと堰の機能の違いを説明してください"},
            {"id": 145, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "上流と下流の管理の違いは何ですか？"},
            {"id": 146, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "構造物対策と非構造物対策の関係を教えてください"},
            {"id": 147, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "計画と設計の関係について説明してください"},
            {"id": 148, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "調査と点検の違いは何ですか？"},
            {"id": 149, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "補修と更新の判断基準の違いを教えてください"},
            {"id": 150, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "治水と利水の関係性を説明してください"},
            {"id": 151, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "砂防と河川の連携について教えてください"},
            {"id": 152, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "ハード対策とソフト対策の組み合わせ方は？"},
            {"id": 153, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "国と地方の役割分担について説明してください"},
            {"id": 154, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "設計と施工の関係を教えてください"},
            {"id": 155, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "短期対策と長期対策の違いは何ですか？"},
            {"id": 156, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "定期点検と臨時点検の使い分けを説明してください"},
            {"id": 157, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "事前防災と事後対応の関係を教えてください"},
            {"id": 158, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "専門家と住民の役割の違いは何ですか？"},
            {"id": 159, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "過去の災害と将来予測の関係を説明してください"},
            {"id": 160, "category": "比較・関係性の説明", "granularity": "coarse",
             "question": "環境保全と治水の両立について教えてください"},
        ])
        
        # 全体構造・体系の理解（20問）
        coarse_questions.extend([
            {"id": 161, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "河川管理の全体フローを説明してください"},
            {"id": 162, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "河川整備の計画体系について教えてください"},
            {"id": 163, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "維持管理のPDCAサイクルを説明してください"},
            {"id": 164, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "技術基準の体系構造を教えてください"},
            {"id": 165, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "河川事業の全体プロセスを説明してください"},
            {"id": 166, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "水系の階層構造について教えてください"},
            {"id": 167, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "点検から補修までの流れを説明してください"},
            {"id": 168, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "調査の体系と種類を教えてください"},
            {"id": 169, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "設計の段階的プロセスを説明してください"},
            {"id": 170, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "河川施設の分類体系を教えてください"},
            {"id": 171, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "災害対応の全体フローを説明してください"},
            {"id": 172, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "情報伝達の仕組みについて教えてください"},
            {"id": 173, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "河川管理者の組織体系を説明してください"},
            {"id": 174, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "許認可の手続きフローを教えてください"},
            {"id": 175, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "予算執行の全体プロセスを説明してください"},
            {"id": 176, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "関係機関の連携体制について教えてください"},
            {"id": 177, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "法体系の階層構造を説明してください"},
            {"id": 178, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "データ管理の全体システムを教えてください"},
            {"id": 179, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "教育・研修の体系を説明してください"},
            {"id": 180, "category": "全体構造・体系の理解", "granularity": "coarse",
             "question": "技術開発の推進体制について教えてください"},
        ])
        
        # 原則・考え方の説明（20問）
        coarse_questions.extend([
            {"id": 181, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "安全性確保の基本原則は何ですか？"},
            {"id": 182, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "環境配慮の基本的な考え方を教えてください"},
            {"id": 183, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "経済性の原則について説明してください"},
            {"id": 184, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "持続可能性の考え方とは何ですか？"},
            {"id": 185, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "リスク管理の基本原則を教えてください"},
            {"id": 186, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "透明性確保の考え方について説明してください"},
            {"id": 187, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "公平性の原則とは何ですか？"},
            {"id": 188, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "予防原則の考え方を教えてください"},
            {"id": 189, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "協働の原則について説明してください"},
            {"id": 190, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "効率性の考え方とは何ですか？"},
            {"id": 191, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "適応性の原則を教えてください"},
            {"id": 192, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "統合的管理の考え方について説明してください"},
            {"id": 193, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "優先順位の考え方とは何ですか？"},
            {"id": 194, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "段階的整備の原則を教えてください"},
            {"id": 195, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "継続性の考え方について説明してください"},
            {"id": 196, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "柔軟性確保の原則とは何ですか？"},
            {"id": 197, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "説明責任の考え方を教えてください"},
            {"id": 198, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "バランスの原則について説明してください"},
            {"id": 199, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "科学的根拠に基づく判断の考え方とは何ですか？"},
            {"id": 200, "category": "原則・考え方の説明", "granularity": "coarse",
             "question": "長期的視点の重要性について教えてください"},
        ])
        
        return coarse_questions
    
    def generate_and_save(self):
        """質問を生成してJSONファイルに保存"""
        logger.info("Generating questions...")
        
        fine_questions = self.generate_fine_grained_questions()
        coarse_questions = self.generate_coarse_grained_questions()
        
        all_questions = fine_questions + coarse_questions
        
        # 統計情報
        fine_count = len([q for q in all_questions if q["granularity"] == "fine"])
        coarse_count = len([q for q in all_questions if q["granularity"] == "coarse"])
        
        dataset = {
            "metadata": {
                "total_questions": len(all_questions),
                "fine_grained": fine_count,
                "coarse_grained": coarse_count,
                "fine_categories": FINE_GRAINED_CATEGORIES,
                "coarse_categories": COARSE_GRAINED_CATEGORIES,
            },
            "questions": all_questions
        }
        
        # JSONファイルに保存
        with open(QUESTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Generated {len(all_questions)} questions")
        logger.info(f"  - Fine-grained: {fine_count}")
        logger.info(f"  - Coarse-grained: {coarse_count}")
        logger.info(f"✓ Saved to {QUESTIONS_FILE}")
        
        return dataset


def main():
    """メイン実行"""
    generator = QuestionGenerator()
    dataset = generator.generate_and_save()
    
    print("\n=== Question Dataset Summary ===")
    print(f"Total: {dataset['metadata']['total_questions']}")
    print(f"Fine-grained: {dataset['metadata']['fine_grained']}")
    print(f"Coarse-grained: {dataset['metadata']['coarse_grained']}")
    print(f"\nSaved to: {QUESTIONS_FILE}")


if __name__ == "__main__":
    main()
