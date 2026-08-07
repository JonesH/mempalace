"""Offline tests for EmbeddinggemmaMLX.

The real MLX model is ~200 MB and pulled from HuggingFace on first use, so
these tests mock mlx_embeddings.load and mlx.core to keep CI fast and
network-free. Skipped when mlx isn't installed (macOS only).
"""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("mlx.core")

import mempalace.embedding as embedding  # noqa: E402  (after importorskip)


@pytest.fixture(autouse=True)
def isolate_embedding_state(monkeypatch):
    monkeypatch.setattr(embedding, "_EF_CACHE", {})
    monkeypatch.setattr(embedding, "_WARNED", set())


class _FakeProcessor:
    def __call__(self, texts, max_length=None, padding=True, truncation=True):
        # Deterministic token ids: one per char, so different texts get
        # different ids and the fake model can be input-dependent. Padded
        # to the longest text like the real tokenizer.
        ids = [[ord(c) % 1000 for c in t] for t in texts]
        maxlen = max(len(i) for i in ids)
        return {
            "input_ids": [i + [0] * (maxlen - len(i)) for i in ids],
            "attention_mask": [[1] * len(i) + [0] * (maxlen - len(i)) for i in ids],
        }


class _FakeModel:
    def __init__(self, out_dim=768):
        self._out_dim = out_dim

    def __call__(self, ids, mask):
        import mlx.core as mx

        batch = ids.shape[0]
        # Deterministic, input-dependent vectors so tests can assert the
        # model output actually flows through (not a constant).
        data = np.arange(batch * self._out_dim, dtype=np.float32).reshape(batch, self._out_dim)
        data = data + np.asarray(ids[:, 0:1], dtype=np.float32) * 0.001
        return type("Out", (), {"text_embeds": mx.array(data)})()


def _install_fakes(monkeypatch, out_dim=768):
    import mlx.core as mx

    monkeypatch.setattr(embedding, "_EF_CACHE", {})
    monkeypatch.setattr(embedding, "_WARNED", set())
    monkeypatch.setattr(
        "mlx_embeddings.load",
        lambda model_id: (_FakeModel(out_dim), _FakeProcessor()),
    )
    # mx.eval on the fake wrapper must be a no-op that returns the array.
    monkeypatch.setattr(mx, "eval", lambda x: x)
    return mx


def test_mlx_ef_embeds_and_truncates(monkeypatch):
    _install_fakes(monkeypatch)
    ef = embedding.EmbeddinggemmaMLX()
    out = ef(["Hallo, das ist ein Test", "zweiter Text"])
    assert len(out) == 2
    assert len(out[0]) == 384  # MRL truncation
    assert len(out[1]) == 384


def test_mlx_ef_normalizes(monkeypatch):
    _install_fakes(monkeypatch)
    ef = embedding.EmbeddinggemmaMLX()
    out = ef(["ein Text"])
    norm = np.linalg.norm(out[0])
    assert abs(norm - 1.0) < 1e-4


def test_mlx_ef_batches(monkeypatch):
    _install_fakes(monkeypatch)
    ef = embedding.EmbeddinggemmaMLX(batch_size=2)
    out = ef([f"text {i}" for i in range(5)])
    assert len(out) == 5
    assert all(len(v) == 384 for v in out)


def test_mlx_ef_empty_and_none(monkeypatch):
    _install_fakes(monkeypatch)
    ef = embedding.EmbeddinggemmaMLX()
    assert ef([]) == []
    assert ef(None) == []
    assert ef("bare string")  # bare string coerced to a single doc


def test_mlx_ef_embed_query_documents(monkeypatch):
    _install_fakes(monkeypatch)
    ef = embedding.EmbeddinggemmaMLX()
    q = ef.embed_query(["query"])
    d = ef.embed_documents(["doc"])
    assert len(q) == 1 and len(d) == 1
    assert len(q[0]) == 384


def test_mlx_ef_lazy_load_once(monkeypatch):
    _install_fakes(monkeypatch)
    calls = []

    monkeypatch.setattr(
        "mlx_embeddings.load",
        lambda model_id: (calls.append(model_id) or _FakeModel(), _FakeProcessor()),
    )
    ef = embedding.EmbeddinggemmaMLX()
    ef(["a"])
    ef(["b"])
    assert len(calls) == 1  # model loaded exactly once


def test_mlx_ef_missing_extra_raises(monkeypatch):
    monkeypatch.setattr(embedding, "_EF_CACHE", {})
    monkeypatch.setattr(embedding, "_WARNED", set())
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlx_embeddings":
            raise ImportError("no mlx-embeddings")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ef = embedding.EmbeddinggemmaMLX()
    with pytest.raises(ImportError, match="mempalace\\[mlx\\]"):
        ef(["text"])


def test_get_embedding_function_mlx_device(monkeypatch):
    _install_fakes(monkeypatch)
    ef = embedding.get_embedding_function(device="mlx", model="embeddinggemma")
    assert isinstance(ef, embedding.EmbeddinggemmaMLX)


def test_get_embedding_function_mlx_cached(monkeypatch):
    _install_fakes(monkeypatch)
    a = embedding.get_embedding_function(device="mlx", model="embeddinggemma")
    b = embedding.get_embedding_function(device="mlx", model="embeddinggemma")
    assert a is b


def test_resolve_providers_mlx(monkeypatch):
    providers, effective = embedding._resolve_providers("mlx")
    assert effective == "mlx"
    assert providers == ["MLXExecutionProvider"]


def test_resolve_providers_mlx_without_mlx(monkeypatch):
    monkeypatch.setattr(embedding, "_WARNED", set())
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlx.core":
            raise ImportError("no mlx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    providers, effective = embedding._resolve_providers("mlx")
    assert effective == "cpu"
    assert providers == ["CPUExecutionProvider"]
