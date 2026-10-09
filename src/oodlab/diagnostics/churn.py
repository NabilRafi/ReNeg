"""Churn diagnostic for TANL's negative-label selection (measurement only).

Attach a :class:`ChurnTracker` to ``run_stream`` as a hook. It never changes what the
method computes (it only reads ``StepOutput.info``, which TANL fills when
``TANLConfig(record=True)``); ``assert_passive`` checks this on a real stream.

Per batch ``t`` with selected set ``S_t`` (the M negatives used for scoring):

* ``churn``          1 - |S_t ∩ S_{t-1}| / M      (``S_{-1}`` = selection at reset)
* ``surv_first``     |S_t ∩ S_first| / M          (``S_first`` = selection of the first batch)
* ``surv_init``      |S_t ∩ S_init| / M           (``S_init`` = selection from ID-text + noise queues)
* ``oracle_overlap`` |S_t ∩ S_oracle| / M         (``S_oracle`` = top-M by the *true* activation
                                                   difference of the segment: mean over its OOD
                                                   images minus mean over its ID images)
* ``unique``         number of distinct labels selected so far
* queue purity       fraction of image entries in the ID (OOD) FIFO queue that are truly ID (OOD),
                     obtained by mirroring the FIFO with ground-truth labels
* admission precision of this batch's admitted samples, and the auto threshold.

Run-level summary: final survival, dwell times (consecutive batches a label stays
selected), batches until the oracle overlap reaches 90 % of its plateau (cold start),
and for temporal streams the same after each shift (post-shift recovery).

Decision rule of the plan: final survival of the first-batch selection above 50 %
supports the cold-start framing; below 10 % means switching to a shift-only framing.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..stream import Stream


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or len(a) == 0:
        return float("nan")
    return len(np.intersect1d(a, b, assume_unique=True)) / float(len(a))


@torch.no_grad()
def oracle_selection(method, stream: Stream, seg_index: int, device="cpu", chunk: int = 1024) -> np.ndarray:
    """Top-M corpus labels by the ground-truth activation difference of one segment."""
    seg = stream.segments[seg_index]
    lab = stream.labels[seg.start:seg.end]
    sum_ood = None
    sum_id = None
    n_ood = int((lab == -1).sum())
    n_id = int((lab != -1).sum())
    for s in range(seg.start, seg.end, chunk):
        e = min(seg.end, s + chunk)
        act = method.activation(stream.feats(s, e, device).to(method.device))
        m = torch.as_tensor(stream.labels[s:e] == -1, device=act.device)
        so, si = act[m].sum(0), act[~m].sum(0)
        sum_ood = so if sum_ood is None else sum_ood + so
        sum_id = si if sum_id is None else sum_id + si
    diff = sum_ood / max(n_ood, 1) - sum_id / max(n_id, 1)
    return method.select(diff).cpu().numpy().astype(np.int32)


class ChurnTracker:
    """Hook for ``run_stream``; see the module docstring for the measured quantities."""

    def __init__(self, oracles: Optional[Dict[int, np.ndarray]] = None, name: str = ""):
        self.oracles = oracles or {}
        self.name = name
        self.records: List[Dict] = []
        self.sel_hist: List[np.ndarray] = []
        self.batch_seg: List[int] = []
        self.init_sel: Dict[int, np.ndarray] = {}
        self.first_sel: Optional[np.ndarray] = None
        self.prev: Optional[np.ndarray] = None
        self.ever: Optional[np.ndarray] = None
        self.resets: List[int] = []
        self.cur_init: Optional[np.ndarray] = None
        self._t = 0

    # ------------------------------------------------------------------ hooks
    def on_reset(self, method, seg: int) -> None:
        if not hasattr(method, "init_selection"):
            raise TypeError("ChurnTracker needs a TANL-like method (init_selection, record=True)")
        L = method.cfg.queue_len
        self.init_sel[seg] = method.init_selection.cpu().numpy().astype(np.int32)
        self.cur_init = self.init_sel[seg]
        self.prev = self.cur_init
        self.first_sel = None
        self.pos_q = deque([2] * int(method.pos_q.shape[0]), maxlen=L)  # 2 = initial (text) entry
        self.neg_q = deque([2] * int(method.neg_q.shape[0]), maxlen=L)  # 2 = initial (noise) entry
        self.ever = np.zeros(method.N, dtype=bool)
        self.ever[self.prev] = True
        self.resets.append(self._t)

    def on_step(self, method, seg: int, s: int, e: int, out, stream: Stream) -> None:
        info = out.info
        if "sel" not in info:
            raise ValueError("TANL must run with TANLConfig(record=True) for the churn tracker")
        sel = info["sel"].numpy().astype(np.int32)
        sel0 = info["sel0"].numpy().astype(np.int32)
        lab = stream.labels[s:e]
        is_ood = lab == -1
        idm = info["id_mask"].numpy().astype(bool)
        oodm = info["ood_mask"].numpy().astype(bool)
        for v in is_ood[idm]:
            self.pos_q.append(0 if v else 1)   # 1 = correct (true ID)
        for v in is_ood[oodm]:
            self.neg_q.append(1 if v else 0)   # 1 = correct (true OOD)
        if self.first_sel is None:
            self.first_sel = sel
        self.ever[sel] = True

        def purity(q):
            img = [c for c in q if c != 2]
            return (sum(img) / len(img)) if img else float("nan"), sum(1 for c in q if c == 2) / max(len(q), 1)

        pos_pur, pos_init = purity(self.pos_q)
        neg_pur, neg_init = purity(self.neg_q)
        rec = {
            "batch": self._t, "segment": seg, "segment_name": stream.segments[seg].name, "start": s, "end": e,
            "n": e - s, "frac_ood": float(is_ood.mean()), "thr": info.get("thr", float("nan")),
            "n_admit_id": int(idm.sum()), "n_admit_ood": int(oodm.sum()),
            "prec_admit_id": float((~is_ood[idm]).mean()) if idm.any() else float("nan"),
            "prec_admit_ood": float(is_ood[oodm].mean()) if oodm.any() else float("nan"),
            "recall_admit_id": float(idm[~is_ood].mean()) if (~is_ood).any() else float("nan"),
            "recall_admit_ood": float(oodm[is_ood].mean()) if is_ood.any() else float("nan"),
            "churn": 1.0 - _overlap(sel, self.prev),
            "churn_sel0_vs_sel": 1.0 - _overlap(sel0, sel),
            "surv_first": _overlap(sel, self.first_sel),
            "surv_init": _overlap(sel, self.cur_init),
            "oracle_overlap": _overlap(sel, self.oracles.get(seg)),
            "unique": int(self.ever.sum()),
            "pos_purity": pos_pur, "neg_purity": neg_pur, "pos_init_frac": pos_init, "neg_init_frac": neg_init,
        }
        for k, orc in self.oracles.items():
            rec[f"overlap_oracle_seg{k}"] = _overlap(sel, orc)
        self.records.append(rec)
        self.sel_hist.append(sel)
        self.batch_seg.append(seg)
        self.prev = sel
        self._t += 1

    # ------------------------------------------------------------------ results
    def frame(self):
        import pandas as pd

        return pd.DataFrame(self.records)

    def dwell_times(self) -> np.ndarray:
        """Lengths (in batches) of every uninterrupted stay of a label in the selected set."""
        if not self.sel_hist:
            return np.array([], dtype=np.int64)
        labels = np.unique(np.concatenate(self.sel_hist))
        col = {int(l): i for i, l in enumerate(labels)}
        T = len(self.sel_hist)
        present = np.zeros((T, len(labels)), dtype=bool)
        for t, s in enumerate(self.sel_hist):
            present[t, [col[int(x)] for x in s]] = True
        runs = []
        for j in range(present.shape[1]):
            x = np.concatenate([[False], present[:, j], [False]]).astype(np.int8)
            d = np.diff(x)
            starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            runs.extend((ends - starts).tolist())
        return np.asarray(runs, dtype=np.int64)

    @staticmethod
    def _batches_to_plateau(x: np.ndarray, frac: float = 0.9, tail: float = 0.25) -> Optional[int]:
        x = np.asarray(x, dtype=float)
        if len(x) == 0 or np.all(np.isnan(x)):
            return None
        plateau = np.nanmedian(x[int(len(x) * (1 - tail)):])
        hit = np.flatnonzero(x >= frac * plateau)
        return int(hit[0]) if len(hit) else None

    def summary(self) -> Dict:
        df = self.frame()
        if df.empty:
            return {}
        out: Dict = {"name": self.name, "batches": int(len(df))}
        seg0 = df[df["segment"] == df["segment"].min()]
        out["final_surv_first"] = float(seg0["surv_first"].iloc[-1])
        out["final_surv_init"] = float(seg0["surv_init"].iloc[-1])
        out["final_oracle_overlap"] = float(seg0["oracle_overlap"].iloc[-1])
        out["batches_to_90pct_oracle_plateau"] = self._batches_to_plateau(seg0["oracle_overlap"].values)
        out["churn_first10"] = float(seg0["churn"].iloc[:10].mean())
        half = len(seg0) // 2
        out["churn_second_half"] = float(seg0["churn"].iloc[half:].mean())
        out["unique_labels"] = int(df["unique"].iloc[-1])
        dw = self.dwell_times()
        if len(dw):
            out.update({"dwell_median": float(np.median(dw)), "dwell_mean": float(dw.mean()),
                        "dwell_p90": float(np.percentile(dw, 90))})
        out["final_pos_purity"] = float(seg0["pos_purity"].iloc[-1])
        out["final_neg_purity"] = float(seg0["neg_purity"].iloc[-1])
        out["mean_prec_admit_id"] = float(seg0["prec_admit_id"].mean())
        out["mean_prec_admit_ood"] = float(seg0["prec_admit_ood"].mean())
        # temporal streams: behaviour after each shift
        shifts = []
        for k in sorted(df["segment"].unique())[1:]:
            sk = df[df["segment"] == k]
            shifts.append({
                "segment": int(k), "name": sk["segment_name"].iloc[0],
                "churn_at_shift": float(sk["churn"].iloc[0]),
                "overlap_at_shift": float(sk["oracle_overlap"].iloc[0]),
                "final_overlap": float(sk["oracle_overlap"].iloc[-1]),
                "batches_to_90pct_plateau": self._batches_to_plateau(sk["oracle_overlap"].values),
            })
        if shifts:
            out["shifts"] = shifts
        out["decision"] = decision(out["final_surv_first"])
        return out


def decision(final_survival: float) -> str:
    if final_survival != final_survival:  # NaN
        return "n/a"
    if final_survival > 0.5:
        return "supports cold-start framing (>50% of first-batch picks survive)"
    if final_survival < 0.1:
        return "first-batch picks wash out (<10%): switch to shift-only framing"
    return "in between (10-50%): discuss before choosing the framing"


def assert_passive(method_plain, method_recording, stream: Stream, device="cpu") -> Dict:
    """Run the same stream with and without recording; scores must be bit-identical."""
    from ..runner import run_stream

    a = run_stream(method_plain, stream, device)
    b = run_stream(method_recording, stream, device, hooks=[ChurnTracker()])
    same = bool(np.array_equal(a.scores, b.scores) and np.array_equal(a.preds, b.preds))
    if not same:
        raise AssertionError("recording changed the scores - the tracker is not passive")
    return {"identical": same, "n": len(a.scores)}


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


def plot_run(df, title: str = "", path: Optional[str] = None, boundaries: Sequence[int] = ()):
    """Six panels: churn, survival, oracle overlap, threshold, queue purity, admission precision."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
    x = df["batch"].values
    ax[0, 0].plot(x, df["churn"], lw=1)
    ax[0, 0].set_title("churn: 1 - |S_t ∩ S_t-1| / M")
    ax[0, 1].plot(x, df["surv_first"], label="first-batch set")
    ax[0, 1].plot(x, df["surv_init"], label="init (text/noise) set")
    ax[0, 1].axhline(0.5, ls="--", c="gray", lw=0.8)
    ax[0, 1].axhline(0.1, ls=":", c="gray", lw=0.8)
    ax[0, 1].set_title("survival")
    ax[0, 1].legend(fontsize=8)
    ax[0, 2].plot(x, df["oracle_overlap"], c="C2")
    ax[0, 2].set_title("overlap with oracle top-M")
    ax[1, 0].plot(x, df["thr"], c="C3")
    ax[1, 0].set_title("automatic threshold")
    ax[1, 1].plot(x, df["pos_purity"], label="ID queue (true ID)")
    ax[1, 1].plot(x, df["neg_purity"], label="OOD queue (true OOD)")
    ax[1, 1].set_title("queue purity (image entries)")
    ax[1, 1].legend(fontsize=8)
    ax[1, 2].plot(x, df["prec_admit_id"], label="admitted as ID")
    ax[1, 2].plot(x, df["prec_admit_ood"], label="admitted as OOD")
    ax[1, 2].set_title("admission precision")
    ax[1, 2].legend(fontsize=8)
    for a in ax.flat:
        a.set_xlabel("batch")
        for b in boundaries:
            a.axvline(b, c="k", lw=0.6, alpha=0.5)
        if a is not ax[1, 0]:
            a.set_ylim(-0.02, 1.02)
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    return fig


def plot_dwell(dwell: np.ndarray, title: str = "", path: Optional[str] = None):
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    if len(dwell):
        ax.hist(dwell, bins=np.arange(1, dwell.max() + 2) - 0.5 if dwell.max() < 80 else 60, log=True)
    ax.set_xlabel("consecutive batches in the selected set")
    ax.set_ylabel("count (log)")
    ax.set_title(title or "dwell time")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    return fig
