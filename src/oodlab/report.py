"""Small helpers the notebooks use to find the cache and to show / save tables."""
from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .cache import FeatureCache, find_existing_caches


def find_cache(ctx, backbone: str = "ViT-B/16", need_textbank: bool = True) -> FeatureCache:
    """The feature cache: /kaggle/working/cache first, then any attached input (notebook 01's output)."""
    roots = [ctx.dirs.cache] + find_existing_caches([ctx.dirs.inputs])
    for r in roots:
        c = FeatureCache(r, backbone)
        if os.path.isdir(c.dir) and (c.has_textbank() or not need_textbank):
            ctx.log.info(f"[cache] using {c.dir}: {c.available()}")
            return c
    raise FileNotFoundError(
        "No feature cache found. Attach the output of notebook 01 (Add Input -> Your Work -> "
        "01-build-feature-cache) or the dataset you made from it. Searched: " + ", ".join(roots))


def show(df, title: Optional[str] = None, cols: Optional[Sequence[str]] = None, digits: int = 2):
    import pandas as pd

    if df is None or len(df) == 0:
        print(f"{title or ''}: (no rows)")
        return
    d = df[[c for c in cols if c in df.columns]] if cols else df   # tolerate absent (e.g. all-NaN) columns
    d = d.round(digits)
    if title:
        print(f"\n=== {title} ===")
    try:
        from IPython.display import display  # type: ignore

        with pd.option_context("display.max_rows", 200, "display.max_columns", 40, "display.width", 250):
            display(d)
    except Exception:  # plain python
        with pd.option_context("display.max_rows", 200, "display.max_columns", 40, "display.width", 250):
            print(d.to_string())


def best_tanl_config(summary_or_path, target: float = 9.81, precision: Optional[str] = None) -> Optional[str]:
    """Name of the TANL configuration ('paper' / 'official_sh') whose Four-OOD mean FPR95 is closest
    to the paper's 9.81, from a summary table (or the tanl_summary.csv written by notebook 02)."""
    import pandas as pd

    t = pd.read_csv(summary_or_path) if isinstance(summary_or_path, str) else summary_or_path
    if t is None or len(t) == 0:
        return None
    t = t[(t["method"] == "tanl") & (t["protocol"] == "four_ood") & (t["key"] == "mean:four")]
    if precision:
        t = t[t["config"].str.endswith("/" + precision)]
    if len(t) == 0:
        return None
    return str(t.loc[(t["fpr95"] - target).abs().idxmin(), "config"]).split("/")[0]


def save_tables(ctx, **tables) -> None:
    for name, df in tables.items():
        if df is not None and len(df):
            df.to_csv(ctx.path(f"{name}.csv"), index=False)


def write_report(ctx, title: str, sections: Iterable[Tuple[str, Union[str, "object"]]], filename: str = "report.md") -> str:
    """Markdown report with tables as fixed-width text (no extra dependencies)."""
    lines: List[str] = [f"# {title}", "", f"code hash `{ctx.info.get('code_hash')}` · device `{ctx.device}` · "
                        f"smoke={ctx.smoke} · {ctx.info.get('started')}", ""]
    for head, body in sections:
        lines += [f"## {head}", ""]
        if hasattr(body, "to_string"):
            lines += ["```", body.round(2).to_string(index=False), "```", ""]
        else:
            lines += [str(body), ""]
    path = ctx.path(filename)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    ctx.log.info(f"report written: {path}")
    return path
