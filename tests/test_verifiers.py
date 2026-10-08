from fgst.verifiers import verify


def test_multiple_choice_requires_final_answer():
    assert verify("openbookqa", "Reasoning. Final answer: C", {"answer": "C"}) == (True, True)
    assert verify("openbookqa", "C", {"answer": "C"}) == (False, False)
    assert verify("openbookqa", "C", {"answer": "C"}, permissive=True) == (True, True)


def test_drop_normalized_exact_or_f1():
    assert verify("drop", "Final answer: New York", {"answer": "New York", "aliases": []}) == (True, True)
