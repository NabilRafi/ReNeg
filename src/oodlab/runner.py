"""Replay test streams through a method and turn the scores into benchmark tables.

    runner = Runner(cache, device="cuda", logger=log)
    df = runner.evaluate(TANL(...), "four_ood", seeds=[0, 1, 2], config="paper")
    summary = summarize(df)
    compare_with_targets(summary, "tanl", "four_ood")

Every row of ``df`` is one (dataset, seed) run with all OpenOOD metrics, plus
"cold-start" metrics on the first ``window`` samples of the stream.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch

from .cache import FeatureCache, FeatureSet
from .methods.base import StreamMethod
from .metrics import openood_metrics
from .stream import TEMPORAL_ORDERS, Segment, Stream, pair_stream, temporal_stream
from .targets import ACCEPTANCE, TARGETS


# ---------------------------------------------------------------------------
# protocols
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Protocol:
    name: str
    id_set: str
    groups: Dict[str, List[str]]
    note: str = ""

    @property
    def ood_sets(self) -> List[str]:
        return [n for g in self.groups.values() for n in g]


PROTOCOLS: Dict[str, Protocol] = {p.name: p for p in [
    Protocol("four_ood", "imagenet_val_all", {"four": ["inaturalist", "sun", "places", "textures_all"]},
             "TANL Tab. 1/A6: MOS Four-OOD, ID = all 50k ImageNet val images"),
    Protocol("openood_v15", "imagenet_test", {"near": ["ssb_hard", "ninco"],
                                              "far": ["inaturalist", "textures", "openimage_o"]},
             "TANL Tab. 2/A7: OpenOOD v1.5, ID = 45k test split"),
    Protocol("near_50k", "imagenet_val_all", {"near": ["ssb_hard", "ninco"]},
             "near-OOD with the 50k ID set (AdaNeg/NegLabel repo logs)"),
    Protocol("openood_val", "imagenet_val", {"val": ["openimage_o_val"]},
             "OpenOOD v1.5 validation split: the only split to tune hyper-parameters on"),
    Protocol("fi_imagenet_1k", "imagenet_val_all_minus_fi", {"near": ["fi_imagenet_1k"]},
             "Fi-ImageNet-1k (derived from cached ImageNet val; needs the authors' lists)"),
]}


# ---------------------------------------------------------------------------
# running a stream
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    scores: np.ndarray
    preds: np.ndarray
    labels: np.ndarray
    segments: List[Segment]
    seconds: float
    n_batches: int
    meta: Dict = field(default_factory=dict)

    def seg_slice(self, k: int) -> slice:
        s = self.segments[k]
        return slice(s.start, s.end)

    def segment_metrics(self, k: int) -> Dict[str, float]:
        sl = self.seg_slice(k)
        return openood_metrics(self.scores[sl], self.labels[sl], self.preds[sl])

    def window_metrics(self, k: int, n_first: int) -> Dict[str, float]:
        """Metrics on the first ``n_first`` samples of segment ``k`` (cold start / post-shift)."""
        s = self.segments[k]
        sl = slice(s.start, min(s.end, s.start + n_first))
        lab = self.labels[sl]
        if (lab == -1).all() or (lab != -1).all():
            return {"fpr95": float("nan"), "auroc": float("nan")}
        m = openood_metrics(self.scores[sl], lab, self.preds[sl])
        return {"fpr95": m["fpr95"], "auroc": m["auroc"]}


def seed_method(method: StreamMethod, seed: int) -> None:
    """Tie a method's own randomness to the stream seed (TANL: the permutation of the initial
    queues, which the official code draws from the unseeded global RNG). One seed = one "run"."""
    cfg = getattr(method, "cfg", None)
    if cfg is not None and hasattr(cfg, "init_seed") and hasattr(cfg, "with_"):
        method.cfg = cfg.with_(init_seed=int(seed))


def run_stream(method: StreamMethod, stream: Stream, device: str = "cpu", hooks: Sequence = ()) -> RunResult:
    """Feed the stream batch by batch. ``hooks`` may define ``on_reset(method, seg)`` and
    ``on_step(method, seg, start, end, out, stream)`` (see diagnostics.churn)."""
    T = len(stream)
    scores = np.empty(T, dtype=np.float64)
    preds = np.empty(T, dtype=np.int64)
    n_b = 0
    t_method = 0.0
    for si, s, e, reset in stream.batches():
        if reset:
            method.reset()
            for h in hooks:
                if hasattr(h, "on_reset"):
                    h.on_reset(method, si)
        x = stream.feats(s, e, device)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = method.step(x)
        sc = out.score.float().cpu().numpy()       # .cpu() synchronises
        pr = out.pred.cpu().numpy()
        t_method += time.perf_counter() - t0
        scores[s:e] = sc
        preds[s:e] = pr
        for h in hooks:
            if hasattr(h, "on_step"):
                h.on_step(method, si, s, e, out, stream)
        n_b += 1
    for h in hooks:
        if hasattr(h, "on_end"):
            h.on_end(method, stream)
    return RunResult(scores, preds, stream.labels.copy(), list(stream.segments), t_method, n_b, dict(stream.meta))


# ---------------------------------------------------------------------------
# protocol evaluation
# ---------------------------------------------------------------------------


class Runner:
    def __init__(self, cache: FeatureCache, device: str = "cpu", batch_size: int = 256, window: int = 2560, logger=None,
                 keep_orders: bool = False):
        self.cache = cache
        self.device = device
        self.batch_size = batch_size
        self.window = window
        self.logger = logger
        self._sets: Dict[str, FeatureSet] = {}
        self.extra_sets: Dict[str, FeatureSet] = {}   # derived sets (e.g. ID minus Fi-ImageNet-1k)
        self.keep_orders = keep_orders
        self.orders: Dict[str, Dict] = {}             # stream orders actually used (see save_orders)

    def _remember(self, key: str, stream: Stream) -> None:
        if self.keep_orders and key not in self.orders:
            self.orders[key] = {"rows": stream.rows.astype(np.int32), "sources": dict(stream.sources),
                                "segments": [(g.name, g.start, g.end, g.reset) for g in stream.segments]}

    def save_orders(self, path: str) -> None:
        """Save every stream order used (rows index the concatenation of ``sources`` in their offset order)."""
        import json

        from .utils import atomic_np_savez

        atomic_np_savez(path, **{k.replace("/", "_"): v["rows"] for k, v in self.orders.items()})
        with open(path.replace(".npz", ".json"), "w") as f:
            json.dump({k: {"sources": v["sources"], "segments": v["segments"]} for k, v in self.orders.items()}, f, indent=1)

    def log(self, msg: str) -> None:
        (self.logger.info if self.logger else print)(msg)

    def get(self, name: str) -> FeatureSet:
        if name in self.extra_sets:
            return self.extra_sets[name]
        if name not in self._sets:
            self._sets[name] = self.cache.load_set(name)
        return self._sets[name]

    def missing(self, protocol: Union[str, Protocol]) -> List[str]:
        p = PROTOCOLS[protocol] if isinstance(protocol, str) else protocol
        have = set(self.cache.available()) | set(self.extra_sets)
        return [n for n in [p.id_set, *p.ood_sets] if n not in have]

    def _row(self, method, config, protocol, group, res: RunResult, k: int, seed, order) -> Dict:
        m = res.segment_metrics(k)
        w = res.window_metrics(k, self.window)
        seg = res.segments[k]
        n = seg.end - seg.start
        return {
            "method": method.name, "config": config, "protocol": protocol, "group": group,
            "dataset": seg.name, "seed": seed, "order": order, **m,
            "cold_fpr95": w["fpr95"], "cold_auroc": w["auroc"],
            "method_seconds": round(res.seconds, 3), "img_per_s": round(n / res.seconds, 1) if res.seconds > 0 else None,
        }

    def evaluate(self, method: StreamMethod, protocol: Union[str, Protocol], seeds: Iterable[int] = (0, 1, 2),
                 config: str = "default", order: str = "shuffled", datasets: Optional[Sequence[str]] = None,
                 hooks_factory: Optional[Callable[[str, int], Sequence]] = None, tag: Optional[str] = None):
        """One row per (OOD set, seed). Memory is reset for every pair, like the official evaluator.

        ``tag`` replaces the protocol name in the output rows (e.g. "order:id_first").
        """
        import pandas as pd

        p = PROTOCOLS[protocol] if isinstance(protocol, str) else protocol
        label = tag or p.name
        miss = self.missing(p)
        if miss:
            raise FileNotFoundError(f"{p.name}: missing cached sets {miss} - run notebook 01 for them")
        id_fs = self.get(p.id_set)
        seeds = list(seeds)
        if getattr(method, "stateless", False) and order == "shuffled":
            seeds = seeds[:1]  # a stateless method gives identical metrics for every stream order
        rows = []
        for group, names in p.groups.items():
            for name in names:
                if datasets is not None and name not in datasets:
                    continue
                ood_fs = self.get(name)
                for seed in seeds:
                    stream = pair_stream(id_fs, ood_fs, seed=seed, batch_size=self.batch_size, order=order).to(self.device)
                    self._remember(f"{p.id_set}|{name}|seed{seed}|{order}", stream)
                    seed_method(method, seed)
                    hooks = hooks_factory(name, seed) if hooks_factory else ()
                    res = run_stream(method, stream, self.device, hooks)
                    row = self._row(method, config, label, group, res, 0, seed, order)
                    rows.append(row)
                    self.log(f"[{method.name}/{config}] {label}:{name} seed={seed}  FPR95 {row['fpr95']:.2f}  "
                             f"AUROC {row['auroc']:.2f}  ACC {row['acc']:.2f}  ({row['img_per_s']} img/s)")
                    del stream
        return pd.DataFrame(rows)

    def tune_on_val(self, make_method: Callable[[object], StreamMethod], values: Sequence, seeds: Iterable[int] = (0,),
                    protocol: str = "openood_val", metric: str = "auroc"):
        """OpenOOD's automatic parameter search (APS): pick the value with the best validation AUROC.

        Mirrors ``OODEvaluator.hyperparam_search`` (ID val 5k + OpenImage-O val 1,763; ties -> the
        last best value). Returns (best_value, table)."""
        import pandas as pd

        rows = []
        for v in values:
            df = self.evaluate(make_method(v), protocol, seeds=seeds, config=f"aps:{v}")
            rows.append({"value": v, metric: float(df[metric].mean()), "fpr95": float(df["fpr95"].mean())})
        table = pd.DataFrame(rows)
        best = table[metric].max()
        best_value = [r["value"] for r in rows if r[metric] == best][-1]
        self.log(f"[aps] {protocol}: best {metric} {best:.2f} at {best_value}")
        return best_value, table

    def evaluate_temporal(self, method: StreamMethod, orders: Optional[Dict[str, List[str]]] = None,
                          id_set: str = "imagenet_val_all", seeds: Iterable[int] = (0,), config: str = "default",
                          id_mode: str = "full", reset_between: bool = False,
                          hooks_factory: Optional[Callable[[str, int], Sequence]] = None):
        """TANL Tab. A15: OOD sets arrive in sequence, no memory reset between them."""
        import pandas as pd

        orders = orders or TEMPORAL_ORDERS
        rows = []
        for oname, names in orders.items():
            missing = [n for n in [id_set, *names] if n not in set(self.cache.available()) | set(self.extra_sets)]
            if missing:
                raise FileNotFoundError(f"temporal {oname}: missing cached sets {missing}")
            for seed in seeds:
                stream = temporal_stream(self.get(id_set), [self.get(n) for n in names], seed=seed,
                                         batch_size=self.batch_size, id_mode=id_mode,
                                         reset_between=reset_between).to(self.device)
                self._remember(f"temporal|{oname}|seed{seed}|{id_mode}", stream)
                seed_method(method, seed)
                hooks = hooks_factory(oname, seed) if hooks_factory else ()
                res = run_stream(method, stream, self.device, hooks)
                for k in range(len(res.segments)):
                    row = self._row(method, config, f"temporal:{oname}", "four", res, k, seed, "temporal")
                    row["position"] = k
                    row["post_shift_fpr95"] = row.pop("cold_fpr95")
                    row["post_shift_auroc"] = row.pop("cold_auroc")
                    rows.append(row)
                self.log(f"[{method.name}/{config}] temporal {oname} seed={seed}: " + "  ".join(
                    f"{r['dataset']} {r['fpr95']:.2f}" for r in rows[-len(res.segments):]))
                del stream
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


METRIC_COLS = ["fpr95", "auroc", "aupr_in", "aupr_out", "fpr95_idpos", "acc"]


def summarize(df):
    """Mean +/- std over seeds, per dataset and per group mean (mean over datasets, then over seeds)."""
    import pandas as pd

    if df is None or len(df) == 0:
        return pd.DataFrame()
    keys = ["method", "config", "protocol"]
    out = []
    for kvals, g in df.groupby(keys, sort=False):
        base = dict(zip(keys, kvals))
        for ds, gd in g.groupby("dataset", sort=False):
            r = dict(base, key=ds, group=gd["group"].iloc[0], n_seeds=gd["seed"].nunique())
            for c in METRIC_COLS:
                r[c] = gd[c].mean()
                r[c + "_std"] = gd[c].std(ddof=0) if len(gd) > 1 else 0.0
            out.append(r)
        for grp, gg in g.groupby("group", sort=False):
            per_seed = gg.groupby("seed")[METRIC_COLS].mean()
            r = dict(base, key=f"mean:{grp}", group=grp, n_seeds=len(per_seed))
            for c in METRIC_COLS:
                r[c] = per_seed[c].mean()
                r[c + "_std"] = per_seed[c].std(ddof=0) if len(per_seed) > 1 else 0.0
            out.append(r)
    return pd.DataFrame(out)


def compare_with_targets(summary, method: str, protocol: str, sources: Optional[Sequence[str]] = None,
                         config: Optional[str] = None):
    """Side-by-side table of our numbers and the published ones."""
    import pandas as pd

    s = summary[(summary["method"] == method) & (summary["protocol"] == protocol)]
    if config is not None:
        s = s[s["config"] == config]
    rows = []
    for t in TARGETS:
        if t.method != method or t.protocol != protocol or (sources and t.source not in sources):
            continue
        for _, r in s[s["key"] == t.key].iterrows():
            # compare like with like: ID-positive published numbers against our ID-positive FPR95
            col = "fpr95_idpos" if t.convention == "id_pos" else "fpr95"
            rows.append({
                "config": r["config"], "key": t.key, "source": t.source, "fpr_convention": t.convention,
                "fpr95": round(r[col], 2), "fpr95_std": round(r[col + "_std"], 2), "target_fpr95": t.fpr95,
                "d_fpr95": None if t.fpr95 is None else round(r[col] - t.fpr95, 2),
                "auroc": round(r["auroc"], 2), "target_auroc": t.auroc,
                "d_auroc": None if t.auroc is None else round(r["auroc"] - t.auroc, 2),
                "acc": round(r["acc"], 2),
            })
    return pd.DataFrame(rows)


def acceptance_table(summary, config_for: Optional[Dict[str, str]] = None):
    """PASS/FAIL for the pre-G1 acceptance criteria that ``summary`` can answer."""
    import pandas as pd

    rows = []
    for method, protocol, key, source, target, tol in ACCEPTANCE:
        s = summary[(summary["method"] == method) & (summary["protocol"] == protocol) & (summary["key"] == key)]
        if config_for and method in config_for:
            s = s[s["config"] == config_for[method]]
        for _, r in s.iterrows():
            d = r["fpr95"] - target
            rows.append({"method": method, "config": r["config"], "protocol": protocol, "key": key,
                         "ours_fpr95": round(r["fpr95"], 2), "target": target, "tolerance": tol,
                         "delta": round(d, 2), "status": "PASS" if abs(d) <= tol else "CHECK"})
    return pd.DataFrame(rows)
