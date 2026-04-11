from __future__ import annotations


def precision_recall_f1(predictions: list, labels: list) -> dict:
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    tp = sum(1 for p, l in zip(predictions, labels) if p == l and p == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_f1(predictions: list, labels: list, classes: list) -> float:
    f1_scores = []
    for cls in classes:
        binary_preds = [1 if p == cls else 0 for p in predictions]
        binary_labels = [1 if l == cls else 0 for l in labels]
        result = precision_recall_f1(binary_preds, binary_labels)
        f1_scores.append(result["f1"])
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def mrr_at_k(ranked_lists: list[list], relevant: list[list], k: int) -> float:
    rr_sum = 0.0
    for ranks, rels in zip(ranked_lists, relevant):
        rr = 0.0
        for i, item in enumerate(ranks[:k], 1):
            if item in rels:
                rr = 1.0 / i
                break
        rr_sum += rr
    return rr_sum / len(ranked_lists) if ranked_lists else 0.0


def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l(hypothesis: str, reference: str) -> float:
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
