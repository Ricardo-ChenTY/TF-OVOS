import numpy as np

from tf_ovcos.metrics import ambiguity_rows, class_aware, compute_mask_metrics


def test_perfect_mask_metrics():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    metrics = compute_mask_metrics(mask, mask)
    assert metrics.iou == 1.0
    assert metrics.mae == 0.0


def test_class_gate_zeroes_overlap_scores_on_wrong_label():
    mask = np.array([[1, 0], [0, 1]], dtype=bool)
    metrics = compute_mask_metrics(mask, mask)
    gated = class_aware(metrics, "frog", "fish")
    assert gated["cIoU"] == 0.0
    assert gated["cMAE"] == 1.0


def test_ambiguity_counts_localized_wrong_label():
    result = ambiguity_rows([(0.8, "frog", "toad"), (0.4, "fish", "fish")])
    assert result["Loc@0.5"] == 0.5
    assert result["Exact@0.5"] == 0.0
    assert result["num_edges"] == 1
