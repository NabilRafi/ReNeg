import os

from .cache import FeatureCache


def find_cache(ctx, backbone="ViT-B/16", need_textbank=True):
    c = FeatureCache(ctx.dirs.cache, backbone)
    if os.path.isdir(c.dir) and (c.has_textbank() or not need_textbank):
        ctx.log.info(f"[cache] using {c.dir}: {c.available()}")
        return c
    raise FileNotFoundError("No feature cache found in " + c.dir)


def show(df, title=None, cols=None, digits=2):
    if title:
        print(f"\n=== {title} ===")
    d = df[list(cols)] if cols else df
    print(d.round(digits).to_string())


def save_tables(ctx, **tables):
    for name, df in tables.items():
        if df is not None and len(df):
            df.to_csv(ctx.path(f"{name}.csv"), index=False)


def write_report(ctx, title, sections, filename="report.md"):
    path = ctx.path(filename)
    with open(path, "w") as f:
        f.write(f"# {title}\n")
    return path
