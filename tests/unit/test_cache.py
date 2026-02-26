import agent.caching_tool.cache as cache_mod
from agent.caching_tool.cache import InMemoryAnswerCache


def test_cache_key_stable_for_pref_order():
    cache = InMemoryAnswerCache(ttl_seconds=3600)
    prefs_a = {"temperature_unit": "celsius", "language": "en"}
    prefs_b = {"language": "en", "temperature_unit": "celsius"}

    cache._set("weather in london", prefs_a, "answer-a")
    assert cache.get("weather in london", prefs_b) == "answer-a"


def test_cache_ttl_expiry(monkeypatch):
    now = {"t": 100.0}
    monkeypatch.setattr(cache_mod.time, "time", lambda: now["t"])

    cache = InMemoryAnswerCache(ttl_seconds=50)
    cache._set("q", None, "a")
    assert cache.get("q") == "a"

    now["t"] = 151.0
    assert cache.get("q") is None
