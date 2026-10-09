"""Published numbers we compare against (FPR95 and AUROC in %).

Sources
* ``tanl_paper``     TANL (CVPR 2026) Tab. 1/A6 (Four-OOD), Tab. 2/A7 (OpenOOD v1.5),
                     Tab. A15 (temporal shift), Tab. A16 (sample order).
* ``adaneg_paper``   AdaNeg numbers as printed in TANL Tab. A6 / Tab. 2.
* ``adaneg_repo``    logs committed in OpenOOD-VLM ``scripts/ood/adaneg/imagenet.sh`` (first run).
* ``neglabel_paper`` NegLabel numbers as printed in TANL Tab. A6 / Tab. 2. NegLabel's own paper
                     uses the ID-positive FPR convention (our ``fpr95_idpos``).
* ``neglabel_repo``  logs in OpenOOD-VLM ``scripts/ood/neglabel/official.sh``.
* ``mcm_paper``      MCM numbers as printed in TANL Tab. A6 / Tab. 2 (ID-positive convention).

Metric convention: OpenOOD (and therefore TANL and the repo logs) treat OOD as the
positive class; FPR95 = fraction of ID samples flagged OOD when 95 % of OOD samples
are caught. MCM/NegLabel papers use ID-positive FPR95 (fraction of OOD accepted at
95 % ID recall). We report both (``fpr95`` and ``fpr95_idpos``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Target:
    method: str
    protocol: str
    key: str                  # dataset name, or "mean:<group>"
    fpr95: Optional[float]
    auroc: Optional[float]
    source: str
    note: str = ""
    convention: str = "ood_pos"   # "ood_pos" (OpenOOD) or "id_pos" (MCM/NegLabel papers)


ID_POSITIVE_SOURCES = {"neglabel_paper", "mcm_paper"}


def _rows(method, protocol, source, values: Dict[str, tuple], note="") -> List[Target]:
    conv = "id_pos" if source in ID_POSITIVE_SOURCES and protocol == "four_ood" else "ood_pos"
    return [Target(method, protocol, k, v[0], v[1], source, note, conv) for k, v in values.items()]


TARGETS: List[Target] = []
# ---------------------------------------------------------------- TANL
TARGETS += _rows("tanl", "four_ood", "tanl_paper", {
    "inaturalist": (0.42, 99.84), "sun": (3.53, 99.07), "places": (21.90, 95.87), "textures_all": (13.38, 97.11),
    "mean:four": (9.81, 97.97)}, "paper: 9.81 +/- 0.01 over 3 stream seeds")
TARGETS += _rows("tanl", "openood_v15", "tanl_paper", {
    "ssb_hard": (62.51, 83.96), "ninco": (57.61, 85.10), "mean:near": (60.06, 84.53),
    "inaturalist": (0.47, 99.83), "textures": (11.22, 97.53), "openimage_o": (39.95, 91.92), "mean:far": (17.21, 96.43)},
    "ID accuracy 66.82")
for order, vals in {
    "I-S-P-T": [(0.45, 99.84), (4.75, 98.77), (22.16, 95.93), (14.91, 96.67), (10.56, 97.80)],
    "S-P-T-I": [(0.46, 99.83), (3.65, 99.04), (21.37, 95.92), (15.09, 96.70), (10.14, 97.87)],
    "P-T-I-S": [(0.45, 99.84), (5.28, 98.73), (21.67, 95.87), (15.33, 96.66), (11.68, 97.87)],
    "T-I-S-P": [(0.47, 99.83), (4.69, 98.77), (21.30, 95.94), (13.51, 97.04), (9.99, 97.89)],
}.items():
    TARGETS += _rows("tanl", f"temporal:{order}", "tanl_paper", dict(zip(
        ["inaturalist", "sun", "places", "textures_all", "mean:four"], vals)), "Tab. A15")
TARGETS += _rows("tanl", "order:id_first", "tanl_paper", {
    "inaturalist": (2.97, None), "sun": (8.35, None), "places": (29.63, None), "textures_all": (24.91, None),
    "mean:four": (16.47, None)}, "Tab. A16")
TARGETS += _rows("tanl", "order:ood_first", "tanl_paper", {
    "inaturalist": (2.74, None), "sun": (9.06, None), "places": (36.00, None), "textures_all": (19.14, None),
    "mean:four": (16.73, None)}, "Tab. A16")
# ---------------------------------------------------------------- AdaNeg
TARGETS += _rows("adaneg", "four_ood", "adaneg_paper", {
    "inaturalist": (0.59, 99.71), "sun": (9.50, 97.44), "places": (34.34, 94.55), "textures_all": (31.27, 94.93),
    "mean:four": (18.92, 96.66)})
TARGETS += _rows("adaneg", "four_ood", "adaneg_repo", {
    "inaturalist": (0.55, 99.71), "sun": (10.62, 97.35), "places": (33.92, 94.17), "textures_all": (26.70, 95.56),
    "mean:four": (17.95, 96.70)}, "repo log, ACC 65.18")
TARGETS += _rows("adaneg", "near_50k", "adaneg_repo", {
    "ssb_hard": (74.26, 74.97), "ninco": (61.23, 78.14), "mean:near": (67.74, 76.55)}, "repo log (50k ID)")
TARGETS += _rows("adaneg", "openood_v15", "adaneg_paper", {"mean:near": (67.51, 76.70), "mean:far": (17.31, 96.43)},
                 "TANL Tab. 2, ID accuracy 67.13")
# ---------------------------------------------------------------- NegLabel
TARGETS += _rows("neglabel", "four_ood", "neglabel_paper", {
    "inaturalist": (1.91, 99.49), "sun": (20.53, 95.49), "places": (35.59, 91.64), "textures_all": (43.56, 90.22),
    "mean:four": (25.40, 94.21)}, "ID-positive FPR convention")
TARGETS += _rows("neglabel", "four_ood", "neglabel_repo", {
    "inaturalist": (1.24, 99.52), "sun": (21.55, 95.46), "places": (38.54, 92.15), "textures_all": (38.54, 91.11),
    "mean:four": (24.97, 94.56)}, "repo log")
TARGETS += _rows("neglabel", "near_50k", "neglabel_repo", {
    "ssb_hard": (77.58, 72.81), "ninco": (60.93, 77.93), "mean:near": (69.25, 75.37)}, "repo log (ID split not stated)")
TARGETS += _rows("neglabel", "openood_v15", "neglabel_paper", {"mean:near": (69.45, 75.18), "mean:far": (23.73, 94.85)},
                 "TANL Tab. 2, ID accuracy 66.82")
# ---------------------------------------------------------------- MCM
TARGETS += _rows("mcm", "four_ood", "mcm_paper", {
    "inaturalist": (32.20, 94.59), "sun": (38.80, 92.25), "places": (46.20, 90.31), "textures_all": (58.50, 86.12),
    "mean:four": (43.93, 90.82)}, "ID-positive FPR convention")
TARGETS += _rows("mcm", "openood_v15", "mcm_paper", {"mean:near": (79.02, 60.11), "mean:far": (68.54, 84.77)},
                 "TANL Tab. 2, ID accuracy 66.28")

# Acceptance checks of the pre-G1 plan: (method, protocol, key, source, target FPR95, tolerance)
ACCEPTANCE = [
    ("tanl", "four_ood", "mean:four", "tanl_paper", 9.81, 1.0),
    ("tanl", "openood_v15", "mean:near", "tanl_paper", 60.06, 1.5),
    ("tanl", "openood_v15", "mean:far", "tanl_paper", 17.21, 1.5),
    ("adaneg", "four_ood", "mean:four", "adaneg_paper", 18.92, 1.0),
    ("adaneg", "openood_v15", "mean:near", "adaneg_paper", 67.51, 1.5),
]


def targets_for(method: str, protocol: str, source: Optional[str] = None) -> List[Target]:
    return [t for t in TARGETS if t.method == method and t.protocol == protocol and (source is None or t.source == source)]
