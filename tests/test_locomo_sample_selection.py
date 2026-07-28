import json

import sentence_transformers


class _NoopSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass


sentence_transformers.SentenceTransformer = _NoopSentenceTransformer

from test_locomo10 import LoCoMoTester


def _sample():
    return {
        "qa": [],
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1": [],
            "session_1_date_time": "2023-05-01",
        },
        "event_summary": {},
        "observation": {},
        "session_summary": {},
    }


def test_selected_sample_keeps_its_original_dataset_index(tmp_path):
    dataset_path = tmp_path / "locomo.json"
    dataset_path.write_text(json.dumps([_sample() for _ in range(10)]))

    tester = object.__new__(LoCoMoTester)
    tester.dataset_path = dataset_path

    selected = tester.load_dataset(sample_indices=[9])

    assert [sample_idx for sample_idx, _ in selected] == [9]
