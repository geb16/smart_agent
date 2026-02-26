import json

from agent.runtime.context_utils import truncate_json


class NonSerializable:
    pass


def test_truncate_json_handles_non_serializable_payload():
    payload = {"obj": NonSerializable(), "ok": True}
    out = truncate_json(payload, max_chars=1000)

    assert isinstance(out, str)
    assert '"obj"' in out
    assert '"ok": true' in out


def test_truncate_json_applies_tail_truncation():
    payload = {"text": "x" * 500}
    expected = json.dumps(payload, ensure_ascii=False, indent=2)[-80:]
    out = truncate_json(payload, max_chars=80)

    assert len(out) <= 80
    assert out == expected
