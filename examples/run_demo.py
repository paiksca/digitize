"""Drive the full CLI pipeline over the synthetic figures and report accuracy.

This doubles as executable documentation of the operating loop a Claude session
would follow. Run after `python tests/synth.py examples`:

    .venv/bin/python examples/run_demo.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIG = [str(ROOT / ".venv/bin/digitize")]


def run(*args):
    r = subprocess.run(DIG + list(args), capture_output=True, text=True)
    if r.returncode != 0:
        print("  ! command failed:", " ".join(args), "\n", r.stderr)
        sys.exit(1)
    return json.loads(r.stdout) if r.stdout.strip() else {}


CONFIGS = {
    "pk": dict(scales=("linear", "log10"),
               series=[("drugA", "#1f77b4"), ("drugB", "#d62728")],
               split=("1.4",), model=("drugA", "exp1")),
    "dose": dict(scales=("log10", "linear"), series=[("compound", "#2ca02c")],
                 model=("compound", "4pl")),
    "line": dict(scales=("linear", "linear"), series=[("signal", "#9467bd")],
                 kind="line"),
}


def calibrate_args(truth, scales):
    args = []
    nx = truth["xticks"] if scales[0] == "log10" else truth["xticks"][:4]
    ny = truth["yticks"] if scales[1] == "log10" else truth["yticks"][:4]
    for d in nx:
        args += ["--x-ref", f'px={round(d["px"])},val={d["val"]}']
    for d in ny:
        args += ["--y-ref", f'py={round(d["px"])},val={d["val"]}']
    return args


def main():
    examples = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "examples"
    for key, cfg in CONFIGS.items():
        truth = json.loads((examples / f"{key}.truth.json").read_text())
        S = str(examples / f"{key}.digitize")
        img = str(examples / f"{key}.png")
        print(f"\n=== {key} ===")
        run("init", img, "--session", S)
        man = ["manifest", "--session", S, "--type", key,
               "--x", f"name=x,scale={cfg['scales'][0]}",
               "--y", f"name=y,scale={cfg['scales'][1]}"]
        for name, color in cfg["series"]:
            man += ["--series", f"name={name},color={color},kind={cfg.get('kind', 'scatter')}"]
        run(*man)
        cal = run("calibrate", "--session", S, *calibrate_args(truth, cfg["scales"]))
        print(f"  calibrate: reprojection RMS = {cal['reprojection_rms_px']:.3f} px")

        for name, color in cfg["series"]:
            tpt = truth["series"][[s["name"] for s in truth["series"]].index(name)]["points"]
            mid = tpt[len(tpt) // 2]
            ex = ["extract", "--session", S, "--series", name]
            if cfg.get("kind") == "line":
                ex += ["--seed-color", color, "--resample", "30"]
            else:
                ex += ["--sample", f'{round(mid["px"])},{round(mid["py"])}']
            if cfg.get("exclude"):
                ex += ["--exclude", cfg["exclude"]]
            if cfg.get("split"):
                ex += ["--split-factor", cfg["split"][0]]
            r = run(*ex)
            print(f"  extract {name}: n={r['n_points']}")

        run("values", "--session", S, "--all")
        vr = run("verify", "--session", S)
        print(f"  verify flags: {vr['flags'] or 'none'}")

        if cfg.get("model"):
            name, model = cfg["model"]
            fr = run("fit", "--session", S, "--series", name, "--model", model)
            d = fr.get("derived", {})
            shown = {k: round(v, 4) for k, v in d.items() if isinstance(v, (int, float))}
            print(f"  fit {name} [{model}]: {shown}  R2={fr.get('r2')}")
        run("export", "--session", S)
    print("\nDone. Inspect overlays under examples/<key>.digitize/overlays/.")


if __name__ == "__main__":
    main()
