import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_sends_balanced_latency_and_assembles_stream():
    import tts
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"ID3" + b"\x00" * 100)

    async with _client(handler) as c:
        r = await tts.synthesize_chunk("Good evening, sir.", api_key="k", voice_id="v", client=c)
    assert seen["url"] == tts.FISH_TTS_URL
    assert seen["auth"] == "Bearer k"
    assert seen["body"] == {"text": "Good evening, sir.", "reference_id": "v",
                            "format": "mp3", "mp3_bitrate": 128, "latency": "balanced"}
    assert r is not None and r.audio.startswith(b"ID3") and len(r.audio) == 103
    assert r.first_byte_sec >= 0 and r.total_sec >= r.first_byte_sec


@pytest.mark.asyncio
async def test_non_200_returns_none():
    import tts
    async with _client(lambda req: httpx.Response(401, content=b"nope")) as c:
        assert await tts.synthesize_chunk("x", api_key="k", voice_id="v", client=c) is None


@pytest.mark.asyncio
async def test_transport_error_returns_none():
    import tts

    def boom(request):
        raise httpx.ConnectError("down")

    async with _client(boom) as c:
        assert await tts.synthesize_chunk("x", api_key="k", voice_id="v", client=c) is None


@pytest.mark.asyncio
async def test_empty_text_or_missing_key_short_circuits():
    import tts
    calls = []

    async with _client(lambda req: calls.append(1) or httpx.Response(200, content=b"x")) as c:
        assert await tts.synthesize_chunk("   ", api_key="k", voice_id="v", client=c) is None
        assert await tts.synthesize_chunk("hi", api_key="", voice_id="v", client=c) is None
    assert calls == []


class _FakeCommunicate:
    seen = {}

    def __init__(self, text, voice, rate="+0%", proxy=None):
        _FakeCommunicate.seen = {"text": text, "voice": voice, "rate": rate, "proxy": proxy}

    async def stream(self):
        yield {"type": "WordBoundary", "offset": 0}
        yield {"type": "audio", "data": b"ID3"}
        yield {"type": "audio", "data": b"\x00" * 10}


@pytest.mark.asyncio
async def test_edge_assembles_audio_chunks(monkeypatch):
    import types
    import tts
    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=_FakeCommunicate))
    r = await tts.synthesize_edge("Good evening, sir.", voice="en-GB-RyanNeural", proxy="http://p")
    assert _FakeCommunicate.seen == {"text": "Good evening, sir.", "voice": "en-GB-RyanNeural",
                                     "rate": "+0%", "proxy": "http://p"}
    assert r is not None and r.audio == b"ID3" + b"\x00" * 10


@pytest.mark.asyncio
async def test_edge_error_or_empty_text_returns_none(monkeypatch):
    import types
    import tts

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("down")

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=Boom))
    assert await tts.synthesize_edge("hi", voice="v") is None
    assert await tts.synthesize_edge("   ", voice="v") is None
