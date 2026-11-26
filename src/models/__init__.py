from .ease import EASE
from .itemknn import ItemKNN
from .popular import Popular


def build_model(name: str, params: dict):
    name = name.lower()
    if name == "popular":
        return Popular(**params)
    if name == "itemknn":
        return ItemKNN(**params)
    if name == "ease":
        return EASE(**params)
    if name == "lightgcn":
        from .lightgcn import LightGCN  # torch는 optional dependency

        return LightGCN(**params)
    raise ValueError(f"unknown model: {name}")
