"""
Generate synthetic telemetry for one or more seeds.

    python generate.py --seeds 0 1 2 3 4 --out data/

Writes data/seed_<n>/site_<X>.csv plus a manifest.json recording the seed,
generator version, full configuration, row counts, base rates, and SHA-256
hashes so every dataset used in the study can be regenerated and verified.
"""
import argparse, dataclasses, hashlib, json, pathlib, platform
import numpy as np, pandas as pd
from telemetry_gen import DEFAULT_CONFIG, GENERATOR_VERSION, generate


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    for seed in args.seeds:
        out = pathlib.Path(args.out) / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "generator_version": GENERATOR_VERSION,
            "seed": seed,
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "config": dataclasses.asdict(DEFAULT_CONFIG),
            "sites": {},
        }
        for name, df in generate(DEFAULT_CONFIG, seed).items():
            path = out / f"site_{name}.csv"
            df.to_csv(path, index=False)
            manifest["sites"][name] = {
                "file": path.name, "sha256": sha256(path),
                "rows": len(df), "devices": int(df.device_id.nunique()),
                "compromised_devices": int(df.groupby("device_id").is_compromised.max().sum()),
                "label_counts": df.label.value_counts().to_dict(),
                "compromised_row_rate": round(float(df.is_compromised.mean()), 4),
            }
            print(f"seed {seed} site {name}: {len(df):,} rows, "
                  f"compromised rate {df.is_compromised.mean():.2%}, "
                  f"degraded rate {(df.label == 'degraded').mean():.2%}")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
