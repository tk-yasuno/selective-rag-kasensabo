"""
河川砂防技術基準ドメイン専門語彙
River and Sabo Technical Standards Domain Vocabulary

河川・砂防・ダム・地すべり・急傾斜地・雪崩対策に特化した専門用語と翻訳辞書を提供
"""
from typing import List, Set

# ストップワード（ノードラベルから除外する語）
STOP_WORDS = {
    # 一般的なストップワード
    '本資料', '本研究', '主な数', '本編', '本章', '本節',
    '設計', '施工', '申し訳', '提示い', 'いただ',
    
    # 追加のストップワード
    '資料', '研究', '報告', '説明', '記載', '図', '表', '写真',
    'ページ', '参照', '以下', '上記', '前述', '後述',
    '場合', '状態', '程度', '箇所', '部分', '全体',
    'こと', 'もの', 'ため', 'など', 'による', 'について',
    '基準', '技術基準', '通達', '指針',
}

# 河川砂防ドメインキーワード（日本語）
KASENSABO_DOMAIN_KEYWORDS = {
    # === 技術基準体系 ===
    '河川砂防技術基準', '調査編', '計画編', '設計編', '維持管理編',
    '基本計画編', '施設配置等計画編',
    '維持管理編（河川編）', '維持管理編（砂防編）', '維持管理編（ダム編）',
    '必須', '標準', '推奨', '例示', '考え方',
    
    # === 河川関連 ===
    # 河川施設
    '堤防', '護岸', '水制', '根固工', '床止め', '床固工',
    '樋門', '樋管', '水門', '閘門', '陸閘', '排水機場',
    '揚水機場', '河道', '河床', '低水路', '高水敷',
    '堤内地', '堤外地', '背水区間', '感潮区間',
    '築堤', '引堤', '河道掘削', '遊水地', '放水路',
    
    # 河川構造物
    '堰', '取水堰', '可動堰', '固定堰', '魚道',
    '樋門樋管', '排水樋管', '伏越', '暗渠',
    '河川橋', '渡河橋', '潜水橋',
    
    # 河川管理
    '河川区域', '河川保全区域', '河川管理施設', '許可工作物',
    '河川台帳', '河川現況台帳', '水系', '本川', '支川',
    '流域', '流域面積', '集水域', '氾濫域',
    
    # 水文・水理
    '計画高水位', '計画高水流量', 'HWL', 'LWL', '平水位',
    '基本高水', '計画高水', '計画洪水流量',
    '洪水', '出水', '渇水', '平水', '豊水',
    '流量', '水位', '流速', '比流量',
    '雨量', '降雨', '降水量', '流出量', '流出解析',
    '確率年', '確率雨量', '確率洪水', '設計洪水',
    '洪水到達時間', '洪水波', '不等流計算', '等流計算',
    '粗度係数', 'マニング式', '限界水深', '常流', '射流',
    
    # === 砂防関連 ===
    # 砂防施設
    '砂防堰堤', '砂防えん堤', '床固工', '遊砂地工', '土石流堆積工',
    '渓流保全工', '護岸工', '水制工',
    '土石流導流工', '土石流流向制御工',
    '山腹工', '山腹緑化工', '山腹基礎工', '山腹斜面工',
    
    # 土砂災害
    '土砂災害', '土砂流出', '土石流', '掃流砂', '浮遊砂',
    '河床変動', '洗掘', '堆積', '土砂生産', '土砂移動',
    '崩壊', '山腹崩壊', '渓岸侵食', '渓床堆積',
    '流木', '巨石', '巨礫', '転石',
    
    # 土砂管理
    '土砂管理', '土砂動態', '土砂収支', '土砂生産量',
    '堆砂', '除石', '掘削', '浚渫',
    '土砂整備計画', '土砂災害防止法',
    
    # === ダム関連 ===
    # ダム型式
    'ダム', '重力式コンクリートダム', '重力式ダム',
    'アーチダム', 'アーチ式ダム', 'ホロージョイントダム',
    'ロックフィルダム', 'アースダム', 'フィルダム',
    '台形CSGダム', '表面遮水壁型ロックフィルダム',
    
    # ダム構造
    'ダム本体', '堤体', '基礎地盤', 'グラウチング',
    '洪水吐き', '常用洪水吐き', '非常用洪水吐き',
    'クレストゲート', 'オリフィスゲート', '放流管',
    '取水設備', '取水塔', '選択取水設備',
    '減勢工', '減勢池', '副ダム', '水叩き',
    
    # ダム管理
    'ダム操作', '放流操作', 'ゲート操作', '流入量予測',
    '洪水調節', '洪水調節容量', '利水容量', '堆砂容量',
    'サーチャージ水位', '常時満水位', '制限水位',
    '貯水位', '貯水量', 'ダム下流河川流量',
    '只管放流', '自然調節方式', 'ゲート操作方式',
    
    # ダム施設の維持管理
    '堤体観測', '漏水量観測', '揚圧力観測', '変位計測',
    'ダム定期検査', 'ダム総合点検', '臨時点検',
    
    # === 地すべり関連 ===
    '地すべり', '地すべり防止区域', '地すべり防止施設',
    '地すべり防止工', '抑制工', '抑止工',
    '地表水排除工', '地下水排除工', '横ボーリング工',
    '集水井工', '排水トンネル工', '排水ボーリング工',
    'アンカー工', '杭工', 'シャフト工', '抑止杭',
    '地すべりブロック', 'すべり面', '移動土塊', '不動地塊',
    '頭部', '末端部', '側部', '滑落崖', '隆起帯',
    
    # === 急傾斜地・雪崩対策 ===
    '急傾斜地', '急傾斜地崩壊', '急傾斜地崩壊防止施設',
    '擁壁工', '法枠工', 'アンカー付法枠工', '抑止杭工',
    '落石防止網', 'ロックシェッド', 'ロックネット',
    '雪崩', '雪崩対策施設', '雪崩予防柵', '雪崩防護柵',
    'スノーシェッド', 'なだれ防護壁', '吹溜式柵',
    
    # === 維持管理共通 ===
    # 点検・調査
    '点検', '定期点検', '臨時点検', '詳細点検', '総合点検',
    '巡視', '巡回', '状態把握', '変状把握',
    '目視点検', '近接目視', '打音検査', '触診',
    '計測', 'モニタリング', '観測', '測量', '測定',
    '非破壊検査', 'コア採取', 'ボーリング調査',
    
    # 健全度評価
    '健全度', '健全度評価', '判定区分', '評価ランク',
    '健全', '機能低下の可能性', '機能低下', '機能停止',
    'A判定', 'B判定', 'C判定', 'S判定',
    '要対策', '要監視', '経過観察', '応急措置',
    
    # 損傷・変状
    '変状', '損傷', '劣化', '変形', 'ひび割れ',
    '亀裂', 'クラック', '剥離', '剥落', '欠損',
    '腐食', '錆', '変色', '漏水', '吸出し',
    '沈下', '隆起', '傾斜', 'ずれ', '段差',
    '開口', '目地開き', '空洞', '洗掘', '浸食',
    
    # 材料・構造
    'コンクリート', '鉄筋コンクリート', '無筋コンクリート',
    'プレストレストコンクリート', '鋼材', '鉄筋',
    '中性化', '塩害', 'アルカリシリカ反応', '凍害',
    '圧縮強度', '引張強度', '付着強度', 'かぶり',
    
    # 対策・補修
    '維持', '修繕', '更新', '補修', '補強', '改築',
    '予防保全', '事後保全', '長寿命化', 'ライフサイクルコスト',
    '応急措置', '恒久対策', '緊急対策', '監視',
    'ひび割れ補修', '断面修復', '表面被覆', '注入工',
    'アンカー補強', '増厚', '巻立て', '増設',
    
    # === 計画・設計共通 ===
    '外力', '設計外力', '荷重', '地震', '耐震',
    '安全率', '安定計算', '構造計算', '照査',
    '設計基準', '技術基準', '示方書', 'ガイドライン',
    'レベル1地震動', 'レベル2地震動',
    
    # === 環境・生態系 ===
    '環境', '自然環境', '生態系', '水環境', '水質',
    '多自然川づくり', '多自然型川づくり', '河川環境',
    '魚類', '底生動物', '植生', '河畔林', '湿地',
    '連続性', '縦断連続性', '横断連続性',
}

# 日本語→英語翻訳辞書（河川砂防専門用語）
KASENSABO_TRANSLATION_DICT = {
    # === 技術基準体系 ===
    '河川砂防技術基準': 'River Sabo Tech Standards',
    '調査編': 'Survey Vol',
    '計画編': 'Planning Vol',
    '設計編': 'Design Vol',
    '維持管理編': 'Maintenance Vol',
    '基本計画編': 'Basic Planning',
    '施設配置等計画編': 'Facility Layout Planning',
    '必須': 'Mandatory',
    '標準': 'Standard',
    '推奨': 'Recommended',
    '例示': 'Example',
    
    # === 河川 ===
    '堤防': 'Levee',
    '護岸': 'Revetment',
    '水制': 'Spur Dike',
    '根固工': 'Bed Protection',
    '床止め': 'Drop Structure',
    '床固工': 'Grade Control',
    '樋門': 'Sluice Gate',
    '樋管': 'Culvert',
    '水門': 'Water Gate',
    '閘門': 'Lock Gate',
    '陸閘': 'Floodgate',
    '排水機場': 'Pump Station',
    '揚水機場': 'Pumping Station',
    '河道': 'River Channel',
    '河床': 'River Bed',
    '低水路': 'Low Water Channel',
    '高水敷': 'Flood Plain',
    '堤内地': 'Protected Area',
    '堤外地': 'Flood Plain Area',
    '背水区間': 'Backwater Section',
    '感潮区間': 'Tidal Section',
    '築堤': 'Embankment',
    '引堤': 'Levee Setback',
    '河道掘削': 'Channel Excavation',
    '遊水地': 'Retarding Basin',
    '放水路': 'Diversion Channel',
    
    # 河川構造物
    '堰': 'Weir',
    '取水堰': 'Intake Weir',
    '可動堰': 'Movable Weir',
    '固定堰': 'Fixed Weir',
    '魚道': 'Fishway',
    '樋門樋管': 'Sluice & Culvert',
    '伏越': 'Siphon',
    '暗渠': 'Covered Channel',
    
    # 水文・水理
    '計画高水位': 'Design HWL',
    '計画高水流量': 'Design Flood',
    'HWL': 'HWL',
    'LWL': 'LWL',
    '平水位': 'Normal Water Level',
    '基本高水': 'Basic Flood',
    '計画高水': 'Design Flood',
    '洪水': 'Flood',
    '出水': 'Freshet',
    '渇水': 'Drought',
    '流量': 'Discharge',
    '水位': 'Water Level',
    '流速': 'Velocity',
    '雨量': 'Rainfall',
    '降雨': 'Precipitation',
    '流出量': 'Runoff',
    '確率年': 'Return Period',
    '確率雨量': 'Probable Rainfall',
    '粗度係数': 'Roughness Coef',
    '不等流計算': 'GVF Calculation',
    '等流計算': 'Uniform Flow',
    '限界水深': 'Critical Depth',
    '常流': 'Subcritical Flow',
    '射流': 'Supercritical Flow',
    
    # === 砂防 ===
    '砂防堰堤': 'Sabo Dam',
    '砂防えん堤': 'Sabo Dam',
    '床固工': 'Check Dam',
    '遊砂地工': 'Sediment Basin',
    '土石流堆積工': 'Debris Basin',
    '渓流保全工': 'Stream Conservation',
    '護岸工': 'Bank Protection',
    '水制工': 'Groin Work',
    '土石流導流工': 'Debris Flow Channel',
    '土石流流向制御工': 'Debris Flow Deflector',
    '山腹工': 'Hillside Work',
    '山腹緑化工': 'Hillside Revegetation',
    '山腹基礎工': 'Hillside Foundation',
    
    # 土砂災害
    '土砂災害': 'Sediment Disaster',
    '土砂流出': 'Sediment Outflow',
    '土石流': 'Debris Flow',
    '掃流砂': 'Bed Load',
    '浮遊砂': 'Suspended Load',
    '河床変動': 'Riverbed Change',
    '洗掘': 'Scouring',
    '堆積': 'Deposition',
    '土砂生産': 'Sediment Production',
    '土砂移動': 'Sediment Transport',
    '崩壊': 'Slope Failure',
    '山腹崩壊': 'Hillside Collapse',
    '渓岸侵食': 'Bank Erosion',
    '流木': 'Driftwood',
    '巨石': 'Boulder',
    '巨礫': 'Large Gravel',
    
    # 土砂管理
    '土砂管理': 'Sediment Management',
    '土砂動態': 'Sediment Dynamics',
    '土砂収支': 'Sediment Budget',
    '堆砂': 'Sedimentation',
    '除石': 'Desiltation',
    '掘削': 'Excavation',
    '浚渫': 'Dredging',
    
    # === ダム ===
    'ダム': 'Dam',
    '重力式コンクリートダム': 'Gravity Dam',
    '重力式ダム': 'Gravity Dam',
    'アーチダム': 'Arch Dam',
    'アーチ式ダム': 'Arch Dam',
    'ホロージョイントダム': 'Hollow Gravity Dam',
    'ロックフィルダム': 'Rockfill Dam',
    'アースダム': 'Earth Dam',
    'フィルダム': 'Fill Dam',
    'CSGダム': 'CSG Dam',
    
    # ダム構造
    'ダム本体': 'Dam Body',
    '堤体': 'Dam Body',
    '基礎地盤': 'Foundation',
    'グラウチング': 'Grouting',
    '洪水吐き': 'Spillway',
    '常用洪水吐き': 'Service Spillway',
    '非常用洪水吐き': 'Emergency Spillway',
    'クレストゲート': 'Crest Gate',
    'オリフィスゲート': 'Orifice Gate',
    '放流管': 'Outlet Conduit',
    '取水設備': 'Intake Facility',
    '取水塔': 'Intake Tower',
    '選択取水設備': 'Selective Intake',
    '減勢工': 'Energy Dissipator',
    '減勢池': 'Stilling Basin',
    '副ダム': 'Secondary Dam',
    '水叩き': 'Apron',
    
    # ダム管理
    'ダム操作': 'Dam Operation',
    '放流操作': 'Release Operation',
    'ゲート操作': 'Gate Operation',
    '流入量予測': 'Inflow Forecast',
    '洪水調節': 'Flood Control',
    '洪水調節容量': 'Flood Control Capacity',
    '利水容量': 'Conservation Storage',
    '堆砂容量': 'Sediment Storage',
    'サーチャージ水位': 'Surcharge Level',
    '常時満水位': 'Normal Pool Level',
    '制限水位': 'Restricted Level',
    '貯水位': 'Reservoir Level',
    '貯水量': 'Storage Volume',
    '只管放流': 'Full Discharge',
    '自然調節方式': 'Natural Regulation',
    
    # === 地すべり ===
    '地すべり': 'Landslide',
    '地すべり防止区域': 'Landslide Prevention Area',
    '地すべり防止施設': 'Landslide Prevention Facility',
    '地すべり防止工': 'Landslide Control Work',
    '抑制工': 'Prevention Work',
    '抑止工': 'Restraining Work',
    '地表水排除工': 'Surface Drainage',
    '地下水排除工': 'Groundwater Drainage',
    '横ボーリング工': 'Horizontal Boring',
    '集水井工': 'Drainage Well',
    '排水トンネル工': 'Drainage Tunnel',
    'アンカー工': 'Anchor Work',
    '杭工': 'Pile Work',
    'シャフト工': 'Shaft Work',
    '抑止杭': 'Restraining Pile',
    'すべり面': 'Sliding Surface',
    '移動土塊': 'Moving Mass',
    '不動地塊': 'Stable Mass',
    '頭部': 'Head',
    '末端部': 'Toe',
    '滑落崖': 'Scarp',
    '隆起帯': 'Bulge',
    
    # === 急傾斜地・雪崩 ===
    '急傾斜地': 'Steep Slope',
    '急傾斜地崩壊': 'Cliff Collapse',
    '擁壁工': 'Retaining Wall',
    '法枠工': 'Slope Frame',
    '落石防止網': 'Rock Fall Net',
    'ロックシェッド': 'Rock Shed',
    'ロックネット': 'Rock Net',
    '雪崩': 'Avalanche',
    '雪崩対策施設': 'Avalanche Facility',
    '雪崩予防柵': 'Avalanche Prevention Fence',
    '雪崩防護柵': 'Avalanche Protection Fence',
    'スノーシェッド': 'Snow Shed',
    '吹溜式柵': 'Snow Drift Fence',
    
    # === 維持管理 ===
    '点検': 'Inspection',
    '定期点検': 'Periodic Inspection',
    '臨時点検': 'Extraordinary Inspection',
    '詳細点検': 'Detailed Inspection',
    '総合点検': 'Comprehensive Inspection',
    '巡視': 'Patrol',
    '巡回': 'Patrol',
    '状態把握': 'Condition Survey',
    '目視点検': 'Visual Inspection',
    '近接目視': 'Close Visual Inspection',
    '打音検査': 'Hammering Test',
    '計測': 'Measurement',
    'モニタリング': 'Monitoring',
    '観測': 'Observation',
    '非破壊検査': 'NDT',
    
    # 健全度評価
    '健全度': 'Soundness',
    '健全度評価': 'Soundness Evaluation',
    '判定区分': 'Rating Category',
    '健全': 'Sound',
    '機能低下の可能性': 'Potential Degradation',
    '機能低下': 'Degraded',
    '機能停止': 'Malfunction',
    'A判定': 'Rating A',
    'B判定': 'Rating B',
    'C判定': 'Rating C',
    '要対策': 'Action Required',
    '要監視': 'Monitoring Required',
    '経過観察': 'Follow-up',
    '応急措置': 'Emergency Repair',
    
    # 損傷・変状
    '変状': 'Anomaly',
    '損傷': 'Damage',
    '劣化': 'Deterioration',
    '変形': 'Deformation',
    'ひび割れ': 'Crack',
    '亀裂': 'Crack',
    'クラック': 'Crack',
    '剥離': 'Delamination',
    '剥落': 'Spalling',
    '欠損': 'Defect',
    '腐食': 'Corrosion',
    '錆': 'Rust',
    '変色': 'Discoloration',
    '漏水': 'Leakage',
    '吸出し': 'Wash Out',
    '沈下': 'Settlement',
    '隆起': 'Uplift',
    '傾斜': 'Tilt',
    'ずれ': 'Displacement',
    '段差': 'Step',
    '開口': 'Opening',
    '空洞': 'Cavity',
    '洗掘': 'Scouring',
    '浸食': 'Erosion',
    
    # 材料
    'コンクリート': 'Concrete',
    '鉄筋コンクリート': 'RC',
    '無筋コンクリート': 'Plain Concrete',
    'プレストレストコンクリート': 'PC',
    '鋼材': 'Steel',
    '鉄筋': 'Rebar',
    '中性化': 'Carbonation',
    '塩害': 'Salt Damage',
    'アルカリシリカ反応': 'ASR',
    '凍害': 'Frost Damage',
    '圧縮強度': 'Compressive Strength',
    'かぶり': 'Cover',
    
    # 対策
    '維持': 'Maintenance',
    '修繕': 'Repair',
    '更新': 'Renewal',
    '補修': 'Repair',
    '補強': 'Reinforcement',
    '改築': 'Reconstruction',
    '予防保全': 'Preventive Maintenance',
    '事後保全': 'Corrective Maintenance',
    '長寿命化': 'Life Extension',
    'ライフサイクルコスト': 'LCC',
    '恒久対策': 'Permanent Measure',
    '緊急対策': 'Emergency Measure',
    '監視': 'Monitoring',
    
    # === 計画・設計 ===
    '外力': 'External Force',
    '設計外力': 'Design Force',
    '荷重': 'Load',
    '地震': 'Earthquake',
    '耐震': 'Seismic Resistance',
    '安全率': 'Safety Factor',
    '安定計算': 'Stability Analysis',
    '構造計算': 'Structural Analysis',
    '照査': 'Verification',
    '設計基準': 'Design Standard',
    '示方書': 'Specifications',
    'レベル1地震動': 'Level 1 Earthquake',
    'レベル2地震動': 'Level 2 Earthquake',
    
    # === 環境 ===
    '環境': 'Environment',
    '自然環境': 'Natural Environment',
    '生態系': 'Ecosystem',
    '水環境': 'Aquatic Environment',
    '水質': 'Water Quality',
    '多自然川づくり': 'Nature-oriented River',
    '河川環境': 'River Environment',
    '魚類': 'Fish',
    '底生動物': 'Benthos',
    '植生': 'Vegetation',
    '河畔林': 'Riparian Forest',
    '連続性': 'Continuity',
    '縦断連続性': 'Longitudinal Continuity',
    '横断連続性': 'Lateral Continuity',
}


def filter_kasensabo_keywords(keywords: List[str], depth: int = 0) -> List[str]:
    """
    河川砂防キーワードをフィルタリングし、ストップワードを除外
    
    Args:
        keywords: キーワードリスト
        depth: ツリーの深さ（0=ROOT, 1=第一層, 2=第二層, ...）
    
    Returns:
        フィルタリング後のキーワードリスト
    """
    filtered = []
    for kw in keywords:
        # ストップワードをスキップ
        if is_stop_word(kw):
            continue
        # 河川砂防ドメインキーワードを優先
        if is_kasensabo_keyword(kw):
            filtered.append(kw)
        # 一定長以上の一般語も許可
        elif len(kw) >= 3 and not kw.isdigit():
            filtered.append(kw)
    
    return filtered


def is_kasensabo_keyword(word: str) -> bool:
    """河川砂防ドメインキーワードかどうかを判定"""
    return word in KASENSABO_DOMAIN_KEYWORDS


def is_stop_word(word: str) -> bool:
    """ストップワードかどうかを判定"""
    return word in STOP_WORDS


def get_priority_keywords_by_depth(depth: int) -> set:
    """
    深さに応じた優先キーワードカテゴリを取得
    
    Args:
        depth: ツリーの深さ
    
    Returns:
        優先すべきキーワードセット
    """
    if depth == 0:
        # ROOT: 技術基準体系の大分類
        return {
            '河川砂防技術基準', '調査編', '計画編', '設計編', '維持管理編',
            '河川', '砂防', 'ダム', '地すべり', '急傾斜地', '雪崩',
        }
    elif depth == 1:
        # 第1層: 各分野の主要施設・概念
        return {
            '堤防', '護岸', '河道', '砂防堰堤', '山腹工',
            'ダム本体', '地すべり防止施設', '擁壁工',
            '点検', '維持管理', '健全度評価',
        }
    elif depth == 2:
        # 第2層: 具体的な技術・手法
        return {
            '水文', '水理', '土砂管理', '洪水調節',
            '損傷', '劣化', '変状', '補修', '補強',
        }
    else:
        # 第3層以降: 詳細な技術用語
        return KASENSABO_DOMAIN_KEYWORDS


# リストからセットに変換（検索の高速化）
_KASENSABO_KEYWORD_SET = set(KASENSABO_DOMAIN_KEYWORDS)
_STOP_WORD_SET = set(STOP_WORDS)


# 互換性のための関数エイリアス
def translate_keyword(keyword: str) -> str:
    """
    日本語キーワードを英語に翻訳（辞書ベース + ローマ字化）
    
    Args:
        keyword: 日本語キーワード
    
    Returns:
        英訳キーワード
    """
    # 既に英数字のみの場合はそのまま返す
    if keyword.isascii():
        return keyword
    
    # 河川砂防辞書から翻訳を取得
    translated = KASENSABO_TRANSLATION_DICT.get(keyword)
    if translated:
        return translated
    
    # pykakasiでローマ字化を試行
    try:
        import pykakasi
        kks = pykakasi.kakasi()
        result = kks.convert(keyword)
        romaji_parts = []
        for item in result:
            if 'hepburn' in item and item['hepburn']:
                romaji_parts.append(item['hepburn'].capitalize())
        
        if romaji_parts:
            return ' '.join(romaji_parts)
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ pykakasi変換エラー ({keyword}): {e}")
    
    # フォールバック: "[JA:...]"形式
    return f"[JA:{keyword[:3]}]"


if __name__ == "__main__":
    # テスト
    print("=== 河川砂防ドメイン語彙辞書 ===")
    print(f"キーワード数: {len(KASENSABO_DOMAIN_KEYWORDS)}")
    print(f"翻訳辞書数: {len(KASENSABO_TRANSLATION_DICT)}")
    print(f"ストップワード数: {len(STOP_WORDS)}")
    
    print("\n=== サンプル翻訳 ===")
    test_keywords = ['堤防', '砂防堰堤', 'ダム', '地すべり', '健全度評価', 'ひび割れ']
    for kw in test_keywords:
        print(f"{kw} -> {translate_keyword(kw)}")
