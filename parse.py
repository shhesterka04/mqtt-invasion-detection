import argparse, re, pathlib as pl, pandas as pd, numpy as np, sys

ATTACK = re.compile(r"bruteforce", re.I)
NORMAL = re.compile(r"normal", re.I)
TIME_FMT = "%m/%d/%Y, %H:%M:%S:%f" 

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_dir",  required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    return ap.parse_args()

def robust_read(csv_path: pl.Path) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path, low_memory=False, dtype=str)
    if "timestamp" not in df.columns:
        print(f"  ! {csv_path.name}: no 'timestamp' column, skip")
        return None

    ts = pd.to_datetime(df["timestamp"], format=TIME_FMT, errors="coerce")
    fail = ts.isna().sum()
    if fail == len(df):
        print(f"  ! {csv_path.name}: every timestamp failed to parse – skip")
        return None
    if fail / len(df) > 0.1:
        print(f"  ! {csv_path.name}: >10 % bad timestamps ({fail})")

    df["timestamp"] = ts.astype("int64") / 1e9        # ns → s
    df = df.dropna(subset=["timestamp"])

    for c in ("mqtt_messagelength", "ip_len"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df

def process_file(path: pl.Path, out_dir: pl.Path):
    stem = path.stem
    label = 1 if ATTACK.search(stem) else 0           # only normal|bruteforce here
    df = robust_read(path)
    if df is None or df.empty:
        return

    df["is_attack"] = label
    for c in ("src_ip","dst_ip","src_port","dst_port","protocol"):
        df[c] = df[c].astype(str)

    flow_cols = ["src_ip","dst_ip","src_port","dst_port","protocol"]
    df["flow_id"] = pd.factorize(df[flow_cols].apply(tuple, axis=1))[0]

    # --- UNIFLOW -------------------------------------------------------------
    num_cols = ["timestamp","mqtt_messagelength","ip_len"]
    agg_map  = {c:["count","mean","std","min","max","sum"] for c in num_cols if c in df.columns}
    uf = df.groupby("flow_id").agg(agg_map)
    uf.columns = ["_".join(t) for t in uf.columns]
    uf = uf.reset_index().merge(df[flow_cols+["flow_id","is_attack"]].drop_duplicates("flow_id"),
                                on="flow_id")
    uf.to_csv(out_dir/f"{stem}_uniflow.csv", index=False)

    # --- BIFLOW --------------------------------------------------------------
    df["pair_key"] = pd.factorize(
        df.apply(lambda r: tuple(sorted((r.src_ip, r.dst_ip,
                                         r.src_port, r.dst_port))), axis=1)
    )[0]
    df["direction"] = (df["src_ip"] < df["dst_ip"]).astype(int)

    parts = []
    for val,suf in [(0,"fwd"),(1,"bwd")]:
        sub = df[df["direction"]==val]
        ag  = sub.groupby("pair_key").agg(agg_map)
        ag.columns = [f"{suf}_{c}_{s}" for c,s in ag.columns]
        parts.append(ag)
    bf = pd.concat(parts, axis=1).fillna(0)
    meta = df.drop_duplicates("pair_key")[["pair_key","protocol","is_attack"]]
    bf = meta.set_index("pair_key").join(bf).reset_index(drop=True)
    bf.to_csv(out_dir/f"{stem}_biflow.csv", index=False)
    print("  ✓ saved", stem)

if __name__ == "__main__":
    cfg = parse_args()
    inp, outp = pl.Path(cfg.in_dir), pl.Path(cfg.out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    for p in inp.glob("*.csv"):
        if ATTACK.search(p.stem) or NORMAL.search(p.stem):
            print("processing", p.stem)
            process_file(p, outp)
    print("Done →", outp)
