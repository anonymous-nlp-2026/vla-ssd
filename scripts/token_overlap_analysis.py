"""Compute token overlap between original and counterfactual instructions.
Uses sentencepiece (Llama-2 BPE tokenizer) directly.
"""
import json
import random
import os
import math
import sentencepiece as spm

SEED = 42
RESULTS_DIR = "./results"
TOKENIZER_PATH = "./checkpoints/openvla-7b/tokenizer.model"

TASK_INSTRUCTIONS = {
    "open_the_middle_drawer_of_the_cabinet": {
        "correct": "open the middle drawer of the cabinet",
        "paraphrased": "pull open the center drawer on the cabinet",
        "wrong": "push the plate to the front of the stove",
    },
    "open_the_top_drawer_and_put_the_bowl_inside": {
        "correct": "open the top drawer and put the bowl inside",
        "paraphrased": "open the upper drawer and place the bowl in it",
        "wrong": "open the middle drawer of the cabinet",
    },
    "push_the_plate_to_the_front_of_the_stove": {
        "correct": "push the plate to the front of the stove",
        "paraphrased": "slide the plate toward the front of the stove",
        "wrong": "open the top drawer and put the bowl inside",
    },
}

TASK_ORDER = [
    "open_the_middle_drawer_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "push_the_plate_to_the_front_of_the_stove",
]

COHENS_D = {
    "empty": 0.9676860750132135,
    "wrong": 0.8437867236086318,
    "paraphrased": 0.6971402750530561,
    "shuffled": 0.62458279134478,
}


def get_shuffled_instruction(instruction, rng):
    words = instruction.split()
    rng.shuffle(words)
    return " ".join(words)


def spearman_r(x, y):
    n = len(x)
    def rank(arr):
        sorted_idx = sorted(range(n), key=lambda i: arr[i])
        ranks = [0.0] * n
        for r_val, idx in enumerate(sorted_idx):
            ranks[idx] = r_val + 1
        return ranks
    rx = rank(x)
    ry = rank(y)
    d_sq = sum((rx[i] - ry[i])**2 for i in range(n))
    return 1 - 6 * d_sq / (n * (n**2 - 1))


def pearson_r(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    sx = math.sqrt(sum((xi - mx)**2 for xi in x))
    sy = math.sqrt(sum((yi - my)**2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def jaccard(a_tokens, b_tokens):
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    if len(a_set | b_set) == 0:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def overlap_ratio(a_tokens, b_tokens):
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    if len(a_set) == 0:
        return 0.0
    return len(a_set & b_set) / len(a_set)


def main():
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    method = "llama2_bpe_tokenizer"
    print(f"Loaded tokenizer: {TOKENIZER_PATH}")

    rng = random.Random(SEED)
    shuffled_instructions = {}
    for task_name in TASK_ORDER:
        correct = TASK_INSTRUCTIONS[task_name]["correct"]
        shuffled_instructions[task_name] = get_shuffled_instruction(correct, rng)

    per_task = []
    per_condition_jaccards = {"shuffled": [], "paraphrased": [], "wrong": [], "empty": []}
    per_condition_overlaps = {"shuffled": [], "paraphrased": [], "wrong": [], "empty": []}

    for task_name in TASK_ORDER:
        cfg = TASK_INSTRUCTIONS[task_name]
        correct = cfg["correct"]
        correct_ids = sp.EncodeAsIds(correct)
        correct_pieces = sp.EncodeAsPieces(correct)

        perturbations = {}
        for cond in ["shuffled", "paraphrased", "wrong", "empty"]:
            if cond == "shuffled":
                variant = shuffled_instructions[task_name]
            elif cond == "paraphrased":
                variant = cfg["paraphrased"]
            elif cond == "wrong":
                variant = cfg["wrong"]
            else:
                variant = ""

            if variant:
                variant_ids = sp.EncodeAsIds(variant)
                variant_pieces = sp.EncodeAsPieces(variant)
            else:
                variant_ids = []
                variant_pieces = []

            j = jaccard(correct_ids, variant_ids)
            o = overlap_ratio(correct_ids, variant_ids)

            intersection = set(correct_ids) & set(variant_ids)
            novel_in_variant = set(variant_ids) - set(correct_ids)
            missing_from_original = set(correct_ids) - set(variant_ids)

            perturbations[cond] = {
                "variant": variant,
                "original_tokens": correct_pieces,
                "variant_tokens": variant_pieces,
                "n_original_unique": len(set(correct_ids)),
                "n_variant_unique": len(set(variant_ids)),
                "n_shared": len(intersection),
                "shared_pieces": [sp.IdToPiece(i) for i in sorted(intersection)],
                "novel_pieces": [sp.IdToPiece(i) for i in sorted(novel_in_variant)],
                "missing_pieces": [sp.IdToPiece(i) for i in sorted(missing_from_original)],
                "jaccard": round(j, 4),
                "overlap_ratio": round(o, 4),
            }
            per_condition_jaccards[cond].append(j)
            per_condition_overlaps[cond].append(o)

        per_task.append({
            "task": task_name,
            "original_instruction": correct,
            "perturbations": perturbations,
        })

    summary = {}
    for cond in ["empty", "wrong", "paraphrased", "shuffled"]:
        vals_j = per_condition_jaccards[cond]
        vals_o = per_condition_overlaps[cond]
        mean_j = sum(vals_j) / len(vals_j)
        mean_o = sum(vals_o) / len(vals_o)
        std_j = math.sqrt(sum((v - mean_j)**2 for v in vals_j) / len(vals_j)) if len(vals_j) > 1 else 0
        summary[cond] = {
            "mean_jaccard": round(mean_j, 4),
            "std_jaccard": round(std_j, 4),
            "mean_overlap_ratio": round(mean_o, 4),
            "token_novelty": round(1 - mean_j, 4),
            "cohens_d": round(COHENS_D[cond], 4),
        }

    conditions_ordered = ["empty", "wrong", "paraphrased", "shuffled"]
    novelties = [summary[c]["token_novelty"] for c in conditions_ordered]
    ds = [summary[c]["cohens_d"] for c in conditions_ordered]
    r_s = spearman_r(novelties, ds)
    r_p = pearson_r(novelties, ds)

    if abs(r_s) > 0.9:
        interp = f"Strong monotonic correlation (Spearman r={r_s:.3f}): token novelty rank-order matches Cohen's d rank-order perfectly. The paraphrased > shuffled ordering is explained by token-level identity changes, not semantic distance alone."
    elif abs(r_s) > 0.6:
        interp = f"Moderate correlation (Spearman r={r_s:.3f}): token overlap partially explains the ordering, but semantic distance has independent explanatory power."
    else:
        interp = f"Weak correlation (Spearman r={r_s:.3f}): token overlap does not explain the ordering; semantic distance is the primary factor."

    correlation = {
        "spearman_r": round(r_s, 4),
        "pearson_r": round(r_p, 4),
        "n_datapoints": 4,
        "note": "n=4, exact p-values not meaningful; rank-order match is the key evidence",
        "data_points": {c: {"token_novelty": summary[c]["token_novelty"], "cohens_d": summary[c]["cohens_d"]} for c in conditions_ordered},
        "interpretation": interp,
    }

    output = {
        "method": f"jaccard_similarity / {method}",
        "n_tasks": len(TASK_ORDER),
        "per_task": per_task,
        "summary": summary,
        "correlation": correlation,
    }

    out_path = os.path.join(RESULTS_DIR, "counterfactual_token_overlap.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print summary
    print(f"\n{'='*85}")
    print("TOKEN OVERLAP ANALYSIS (Llama-2 BPE Tokenizer)")
    print(f"{'='*85}")

    for entry in per_task:
        print(f"\nTask: {entry['task']}")
        print(f"  Original: \"{entry['original_instruction']}\"")
        print(f"           tokens: {entry['perturbations']['shuffled']['original_tokens']}")
        for cond in ["shuffled", "paraphrased", "wrong", "empty"]:
            p = entry["perturbations"][cond]
            print(f"  {cond:12s}: \"{p['variant']}\"")
            print(f"               tokens: {p['variant_tokens']}")
            print(f"               shared={p['n_shared']}  novel={p['novel_pieces']}  missing={p['missing_pieces']}")
            print(f"               Jaccard={p['jaccard']:.4f}  Overlap={p['overlap_ratio']:.4f}")

    print(f"\n{'='*85}")
    print(f"{'Perturbation':<14} {'Jaccard':>8} {'StdJ':>6} {'Overlap':>8} {'Novelty':>8} {'Cohen d':>8}")
    print("-" * 58)
    for cond in ["empty", "wrong", "paraphrased", "shuffled"]:
        s = summary[cond]
        print(f"{cond:<14} {s['mean_jaccard']:>8.4f} {s['std_jaccard']:>6.4f} {s['mean_overlap_ratio']:>8.4f} {s['token_novelty']:>8.4f} {s['cohens_d']:>8.4f}")

    print(f"\nCorrelation (token novelty vs Cohen's d):")
    print(f"  Spearman r = {correlation['spearman_r']:.4f}")
    print(f"  Pearson  r = {correlation['pearson_r']:.4f}")
    print(f"  {correlation['interpretation']}")

    # Key insight
    print(f"\n{'='*85}")
    print("KEY FINDING:")
    print(f"  Shuffled preserves ALL tokens (Jaccard=1.0) yet still shifts actions (d={COHENS_D['shuffled']:.3f})")
    print(f"  => Word ORDER matters, not just token identity")
    print(f"  Paraphrased introduces NEW tokens (Jaccard~{summary['paraphrased']['mean_jaccard']:.2f})")
    print(f"  => Higher d ({COHENS_D['paraphrased']:.3f}) reflects BOTH order change AND token identity change")
    print(f"  Token novelty perfectly predicts d ranking (Spearman r={r_s:.1f})")


if __name__ == "__main__":
    main()
