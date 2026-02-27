#!/usr/bin/env python3
"""Run all 3 ShopLens evaluation pipelines and produce a consolidated report."""

from __future__ import annotations

import importlib
import json
import os
import sys
import time

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)

from dotenv import load_dotenv
load_dotenv()

from shoplens.report import console, print_summary


def _run(module_name: str):
    """Import a script module by filename (without .py) and call main()."""
    # Scripts live next to this file
    script_dir = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(script_dir, f"{module_name}.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main()


def main() -> None:
    console.print("\n[bold blue]ShopLens Full Evaluation Suite[/bold blue]\n")
    start = time.time()

    all_results: dict[str, dict] = {}

    # Pipeline 1: Background model comparison
    bg = _run("01_compare_bg_models")
    all_results["Background Studio"] = {
        "score": bg["score"],
        "status": "ok",
        "details": f"Winner: {bg['winner']}",
    }

    # Pipeline 2: Enhancer regression check
    enh = _run("02_check_enhancer")
    all_results["Product Enhancer"] = {
        "score": enh["score"],
        "status": enh["status"],
        "details": f"{len(enh['regressions'])} regressions" if enh["regressions"] else "No regressions",
    }

    # Pipeline 3: BG removal quality gate
    bgr = _run("03_eval_bg_removal")
    all_results["Clean Cut"] = {
        "score": bgr["score"],
        "status": bgr["status"],
        "details": f"Flagged: {', '.join(bgr['flagged'])}" if bgr["flagged"] else "All passed",
    }

    # Consolidated summary
    elapsed = time.time() - start
    print_summary(all_results)
    console.print(f"  Completed in {elapsed:.0f}s\n")

    # Save results
    output = {
        "pipelines": {
            "bg_generator": bg,
            "product_enhancer": enh,
            "clean_cut": bgr,
        },
        "elapsed_seconds": round(elapsed, 1),
    }
    results_path = os.path.join(_root, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"  Results saved to results.json\n")


if __name__ == "__main__":
    main()
