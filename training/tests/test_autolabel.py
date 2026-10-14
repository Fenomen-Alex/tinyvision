from pathlib import Path

from tv.data.autolabel import (
    autolabel_source,
    build_prompt,
    parse_label_response,
)


class FakeCaller:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, endpoint, model, prompt, data_urls, timeout):
        self.calls += 1
        return self.replies.pop(0)


def test_parse_plain_json():
    r = parse_label_response('{"label": "failing", "confidence": 0.9, "reason": "blob"}')
    assert r["label"] == "failing"
    assert r["confidence"] == 0.9


def test_parse_json_in_prose():
    r = parse_label_response('Sure!\n```json\n{"label": "normal"}\n```')
    assert r["label"] == "normal"


def test_parse_bad_label_is_uncertain():
    r = parse_label_response('{"label": "banana"}')
    assert r["label"] == "uncertain"


def test_parse_garbage_is_uncertain():
    assert parse_label_response("no json here")["label"] == "uncertain"


def test_build_prompt_mentions_classes():
    p = build_prompt()
    assert "failing" in p and "normal" in p and "uncertain" in p


def test_autolabel_source_aggregates(tmp_path):
    from PIL import Image

    frames = []
    for i in range(3):
        p = tmp_path / f"frame_{i:04d}.jpg"
        Image.new("RGB", (32, 32), (i, i, i)).save(p)
        frames.append(p)
    caller = FakeCaller([
        '{"label": "failing", "confidence": 0.8}',
        '{"label": "failing", "confidence": 0.7}',
        '{"label": "normal", "confidence": 0.6}',
    ])
    result = autolabel_source("abc", frames, caller=caller,
                              sleep=lambda s: None)
    assert result["id"] == "abc"
    assert result["video_label"] == "failing"
    assert result["counts"]["failing"] == 2
    assert result["counts"]["normal"] == 1
    assert len(result["frames"]) == 3


def test_autolabel_source_all_uncertain(tmp_path):
    from PIL import Image

    p = tmp_path / "frame_0000.jpg"
    Image.new("RGB", (32, 32), (0, 0, 0)).save(p)
    caller = FakeCaller(["not json"])
    result = autolabel_source("abc", [p], caller=caller, sleep=lambda s: None)
    assert result["video_label"] == "uncertain"
