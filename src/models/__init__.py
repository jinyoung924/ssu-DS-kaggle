from .affinity import Affinity
from .blend import Blend
from .ease import EASE
from .fileorder import FileOrderNeighbor
from .itemknn import ItemKNN
from .popular import Popular
from .poscooc import PositionCooc
from .precomputed import PrecomputedEmb
from .rp3beta import RP3beta
from .textsim import TextSim
from .timeaware import AliveWindow, EASETime

MODELS = {
    "popular": Popular,
    "itemknn": ItemKNN,
    "ease": EASE,
    "rp3beta": RP3beta,
    "fileorder": FileOrderNeighbor,  # Ordering Artifact (공지사항)
    "poscooc": PositionCooc,         # Ordering Artifact 기반 아이템 유사도
    "affinity": Affinity,            # 브랜드/카테고리 선호
    "textsim": TextSim,              # 상품 텍스트 유사도
    "alive": AliveWindow,            # 유저 활동 기간 안에 팔린 아이템
    "easetime": EASETime,            # 시간 감쇠 EASE
    "precomputed": PrecomputedEmb,   # kaggle/에서 GPU로 학습한 임베딩
    "blend": Blend,
}


def build_model(name: str, params: dict):
    name = name.lower()
    if name == "lightgcn":
        from .lightgcn import LightGCN  # torch는 optional dependency

        return LightGCN(**params)
    if name not in MODELS:
        raise ValueError(f"unknown model: {name}")
    return MODELS[name](**params)
