"""analyze_goal_probes.py — Compare goal classification probe results across trained/untrained/dinov2.

Reads summary.json from each model's goal_classification output directory.
Outputs: per-layer accuracy comparison, best layer summary, gate judgment.

Gates:
  Sanity-GC: DINO best-layer accuracy ≤ 25%
  G1-GC:     trained accuracy > 40%
  G2-GC:     Δ(trained - DINO) > 25pp
  G3-GC:     Δ(trained - untrained) > 10pp
"""

import argparse
import json
import os
import sys


def load_summary(result_dir):
    path = os.path.join(result_dir, "goal_classification", "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description="Analyze goal classification probe results")
    p.add_argument("--trained_dir", required=True, help="Results dir for trained VLA")
    p.add_argument("--untrained_dir", default=None, help="Results dir for untrained VLA")
    p.add_argument("--dino_dir", default=None, help="Results dir for DINO-V2")
    p.add_argument("--acc_gate", type=float, default=0.40, help="Min trained accuracy for G1-GC (default: 0.40)")
    p.add_argument("--gap_gate", type=float, default=0.25, help="Min trained-DINO accuracy gap for G2-GC (default: 0.25)")
    p.add_argument("--sanity_gate", type=float, default=0.25, help="Max DINO accuracy for Sanity-GC (default: 0.25)")
    p.add_argument("--g3_gate", type=float, default=0.10, help="Min trained-untrained accuracy gap for G3-GC (default: 0.10)")
    args = p.parse_args()

    summaries = {}
    for name, path in [("trained", args.trained_dir),
                        ("untrained", args.untrained_dir),
                        ("dinov2", args.dino_dir)]:
        if path is None:
            continue
        s = load_summary(path)
        if s is None:
            print(f"WARNING: No goal_classification/summary.json in {path}")
            continue
        summaries[name] = s

    if not summaries:
        print("No results found. Exiting.")
        sys.exit(1)

    # Per-layer accuracy table
    print("\n" + "=" * 70)
    print("Goal Classification Probe — Per-Layer Accuracy")
    print("=" * 70)

    all_layers = set()
    for s in summaries.values():
        all_layers.update(s.get("accuracy_by_layer", {}).keys())
        for k in s.get("results", {}):
            all_layers.add(str(k))

    def get_acc(summary, layer_key):
        abl = summary.get("accuracy_by_layer", {})
        if layer_key in abl:
            return abl[layer_key]
        res = summary.get("results", {})
        if layer_key in res:
            return res[layer_key].get("accuracy")
        return None

    def get_traj_acc(summary, layer_key):
        tabl = summary.get("traj_accuracy_by_layer", {})
        if layer_key in tabl:
            return tabl[layer_key]
        res = summary.get("results", {})
        if layer_key in res:
            return res[layer_key].get("traj_accuracy")
        return None

    # Sort layers numerically
    sorted_layers = sorted(all_layers, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0))
    model_names = [n for n in ["trained", "untrained", "dinov2"] if n in summaries]

    col_headers = []
    for n in model_names:
        col_headers.append(f"{n}_acc")
        col_headers.append(f"{n}_traj")
    header = f"{'Layer':>6}" + "".join(f"  {h:>12}" for h in col_headers)
    print(header)
    print("-" * len(header))
    for layer in sorted_layers:
        row = f"{layer:>6}"
        for name in model_names:
            acc = get_acc(summaries[name], layer)
            if acc is not None:
                row += f"  {acc:>11.4f}"
            else:
                row += f"  {'—':>11}"
            tacc = get_traj_acc(summaries[name], layer)
            if tacc is not None:
                row += f"  {tacc:>11.4f}"
            else:
                row += f"  {'—':>11}"
        print(row)

    # Best layer summary
    print("\n" + "=" * 70)
    print("Best Layer Summary")
    print("=" * 70)
    for name in model_names:
        s = summaries[name]
        best_layer = s.get("best_layer", "?")
        best_acc = s.get("best_accuracy", 0)
        best_f1 = s.get("best_macro_f1", 0)
        best_tacc = s.get("best_traj_accuracy", 0) or 0
        print(f"  {name:>10}: layer {best_layer}  acc={best_acc:.4f}  traj_acc={best_tacc:.4f}  F1={best_f1:.4f}")

    # Gate judgment
    trained_acc = summaries.get("trained", {}).get("best_accuracy", 0)
    dino_acc = summaries.get("dinov2", {}).get("best_accuracy", 0) if "dinov2" in summaries else None
    untrained_acc = summaries.get("untrained", {}).get("best_accuracy", 0) if "untrained" in summaries else None

    gap_trained_dino = (trained_acc - dino_acc) if dino_acc is not None else None
    gap_trained_untrained = (trained_acc - untrained_acc) if untrained_acc is not None else None

    print("\n" + "=" * 70)
    print("=== Goal Classification Gate Judgment ===")
    print("=" * 70)

    gate_results = []

    # Sanity-GC: DINO best-layer accuracy ≤ 25%
    if dino_acc is not None:
        sanity_pass = dino_acc <= args.sanity_gate
        status = "PASS" if sanity_pass else "FAIL"
        msg = f"Sanity-GC: {status} — DINO best acc = {dino_acc:.1%} {'≤' if sanity_pass else '>'} {args.sanity_gate:.0%}"
        if not sanity_pass:
            msg += ", visual ambiguity not confirmed"
        gate_results.append((status, msg))
    else:
        gate_results.append(("SKIP", "Sanity-GC: SKIP — DINO results not available"))

    # G1-GC: trained accuracy > 40%
    g1_pass = trained_acc > args.acc_gate
    status = "PASS" if g1_pass else "FAIL"
    gate_results.append((status, f"G1-GC:     {status} — Trained best acc = {trained_acc:.1%} {'>' if g1_pass else '≤'} {args.acc_gate:.0%}"))

    # G2-GC: Δ(trained - DINO) > 25pp
    if gap_trained_dino is not None:
        g2_pass = gap_trained_dino > args.gap_gate
        status = "PASS" if g2_pass else "FAIL"
        gate_results.append((status, f"G2-GC:     {status} — Δ(trained-DINO) = {gap_trained_dino:.1%}pp {'>' if g2_pass else '≤'} {args.gap_gate:.0%}pp"))
    else:
        gate_results.append(("SKIP", "G2-GC:     SKIP — DINO results not available"))

    # G3-GC: Δ(trained - untrained) > 10pp
    if gap_trained_untrained is not None:
        g3_pass = gap_trained_untrained > args.g3_gate
        status = "PASS" if g3_pass else "FAIL"
        gate_results.append((status, f"G3-GC:     {status} — Δ(trained-untrained) = {gap_trained_untrained:.1%}pp {'>' if g3_pass else '≤'} {args.g3_gate:.0%}pp"))
    else:
        gate_results.append(("SKIP", "G3-GC:     SKIP — untrained results not found"))

    for _, msg in gate_results:
        print(f"  {msg}")

    n_pass = sum(1 for s, _ in gate_results if s == "PASS")
    n_eval = sum(1 for s, _ in gate_results if s != "SKIP")
    n_total = len(gate_results)
    print(f"  Overall:   {n_pass}/{n_eval} PASS ({n_total - n_eval} skipped)")

    all_passed = all(s == "PASS" for s, _ in gate_results if s != "SKIP")
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
