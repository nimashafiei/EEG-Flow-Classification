from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Tuple, Union

@dataclass
class EEGFlowConfig:
    base_path: str = "Data/XData"
    segment: str = "seg1"
    entropies: Tuple[str, ...] = ("XSpecEn", "XDistEn")
    rows_per_subject: int = 18
    seed: int = 42
    use_rest_center: bool = True
    weights: Optional[Union[Dict[str, float], Sequence[float]]] = None
    threshold: float = 0.5
