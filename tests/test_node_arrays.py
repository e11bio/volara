import json

import numpy as np
import pytest

from volara.node_arrays import NodeArrays


def store(tmp_path, n=50, k=4, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(100, 100 + 3 * n, 3)).astype(np.uint64)
    arrays = {
        "embedding": rng.normal(size=(n, k)).astype(np.float32),
        "count": rng.integers(0, 9, size=n).astype(np.int32),
    }
    nodes = NodeArrays(path=tmp_path / "nodes.arrays")
    nodes.save(ids, arrays, graph_digest="abc", source="model@1")
    return nodes, ids, arrays


def test_save_writes_the_documented_files(tmp_path):
    nodes, ids, arrays = store(tmp_path)
    folder = nodes.local
    assert sorted(p.name for p in folder.iterdir()) == [
        "ids.npy",
        "meta.json",
        "node_count.npy",
        "node_embedding.npy",
        "order.npy",
    ]
    assert json.loads((folder / "meta.json").read_text()) == {
        "version": 1,
        "num_nodes": 50,
        "attrs": {
            "embedding": {"dtype": "<f4", "size": 4},
            "count": {"dtype": "<i4", "size": 0},
        },
        "graph_digest": "abc",
        "source": "model@1",
    }
    # plain npy files: numpy reads them without this package
    stored_ids = np.load(folder / "ids.npy")
    assert stored_ids.dtype == np.dtype("<u8")
    np.testing.assert_array_equal(stored_ids, ids)
    order = np.load(folder / "order.npy")
    assert order.dtype == np.dtype("<i8")
    np.testing.assert_array_equal(order, np.argsort(ids, kind="stable"))
    np.testing.assert_array_equal(
        np.load(folder / "node_embedding.npy"), arrays["embedding"]
    )
    np.testing.assert_array_equal(np.load(folder / "node_count.npy"), arrays["count"])


def test_read_returns_rows_in_the_asked_order(tmp_path):
    nodes, ids, arrays = store(tmp_path)
    asked = ids[[7, 3, 3, 41]]
    found, values = nodes.read(asked)
    assert found.tolist() == [True, True, True, True]
    np.testing.assert_array_equal(
        values["embedding"], arrays["embedding"][[7, 3, 3, 41]]
    )
    np.testing.assert_array_equal(values["count"], arrays["count"][[7, 3, 3, 41]])


def test_read_marks_ids_that_are_not_stored(tmp_path):
    nodes, ids, arrays = store(tmp_path)
    asked = np.array([ids[0], 1, 10**12, ids[5] + 1], dtype=np.uint64)
    found, values = nodes.open().read(asked, ["embedding", "count"])
    assert found.tolist() == [True, False, False, False]
    np.testing.assert_array_equal(values["embedding"][0], arrays["embedding"][0])
    assert np.isnan(values["embedding"][1:]).all()
    assert values["count"][1:].tolist() == [0, 0, 0]


def test_read_of_no_ids_keeps_the_vector_shape(tmp_path):
    nodes, _, _ = store(tmp_path)
    found, values = nodes.read(np.zeros(0, dtype=np.uint64), ["embedding"])
    assert found.shape == (0,)
    assert values["embedding"].shape == (0, 4)


def test_writer_appends_blocks(tmp_path):
    nodes = NodeArrays(path=tmp_path / "nodes.arrays")
    with nodes.writer({"score": ("float32", 0), "pair": ("float64", 2)}) as writer:
        writer.append(
            np.array([5, 1]), score=np.array([0.5, 0.1]), pair=np.ones((2, 2))
        )
        writer.append(np.array([3]), score=np.array([0.3]), pair=np.zeros((1, 2)))
        assert not nodes.exists()  # nothing in place until the block ends
    reader = nodes.open()
    assert len(reader) == 3
    assert reader.ids.tolist() == [5, 1, 3]
    found, values = reader.read(np.array([1, 3, 5]))
    assert found.all()
    np.testing.assert_allclose(values["score"], [0.1, 0.3, 0.5])
    assert values["score"].dtype == np.float32
    np.testing.assert_array_equal(values["pair"], [[1, 1], [0, 0], [1, 1]])


def test_writer_that_fails_leaves_the_earlier_store(tmp_path):
    nodes, ids, arrays = store(tmp_path)
    with pytest.raises(RuntimeError, match="stop"):
        with nodes.writer({"embedding": ("float32", 4)}) as writer:
            writer.append(np.array([1]), embedding=np.zeros((1, 4)))
            raise RuntimeError("stop")
    assert [p.name for p in tmp_path.iterdir()] == ["nodes.arrays"]
    np.testing.assert_array_equal(nodes.open().ids, ids)


def test_save_replaces_an_earlier_store(tmp_path):
    nodes, _, _ = store(tmp_path)
    nodes.save(np.array([2, 1]), {"other": np.array([20.0, 10.0])})
    reader = nodes.open()
    assert list(reader.attrs) == ["other"]
    assert reader.read(np.array([1]))[1]["other"].tolist() == [10.0]


def test_an_id_written_twice_is_refused(tmp_path):
    nodes = NodeArrays(path=tmp_path / "nodes.arrays")
    with pytest.raises(ValueError, match="more than once"):
        nodes.save(np.array([4, 4]), {"score": np.array([1.0, 2.0])})
    assert not nodes.exists()


@pytest.mark.parametrize(
    "arrays, match",
    [
        ({"score": np.zeros(3)}, "shape"),
        ({"other": np.zeros(2)}, "expected the attributes"),
    ],
)
def test_writer_refuses_rows_that_do_not_fit(tmp_path, arrays, match):
    nodes = NodeArrays(path=tmp_path / "nodes.arrays")
    with pytest.raises(ValueError, match=match):
        with nodes.writer({"score": ("float32", 0)}) as writer:
            writer.append(np.array([1, 2]), **arrays)


def test_unknown_attribute_and_missing_store(tmp_path):
    nodes, ids, _ = store(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        nodes.read(ids[:1], ["nope"])
    nodes.drop()
    assert not nodes.exists()
    with pytest.raises(FileNotFoundError):
        nodes.open()
