from fgst.metrics import paired_metrics


def test_paired_transition_identity_and_denominators():
    rows = [
        {"base_success": False, "adapted_success": True, "transition": "rescue"},
        {"base_success": True, "adapted_success": False, "transition": "harm"},
        {"base_success": True, "adapted_success": True, "transition": "success_preserved"},
        {"base_success": False, "adapted_success": False, "transition": "failure_persisted"},
    ]
    result = paired_metrics(rows)
    assert result["rescue"] == result["harm"] == 1
    assert result["net_gain"] == 0
    assert result["conditional_rescue"] == result["conditional_harm"] == 0.5
