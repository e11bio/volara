import json
import textwrap
from pathlib import Path

import daisy
import numpy as np
import pytest
import torch
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import prepare_ds

from volara.blockwise import (
    LUT,
    AffAgglom,
    Argmax,
    DistanceAgglom,
    ExtractFrags,
    GlobalMWS,
    Predict,
    SeededExtractFrags,
)
from volara.datasets import Affs, Labels, Raw
from volara.dbs import SQLite
from volara.models import Checkpoint


def build_configs(tmpdir):
    tmpdir = Path(tmpdir)
    zarr_dir = tmpdir / "out.zarr"
    prepare_ds(
        zarr_dir / "raw",
        shape=(1, 100, 100, 100),
        voxel_size=Coordinate(1, 1, 1),
        offset=Coordinate(0, 0, 0),
        dtype=np.float32,
        mode="w",
    )
    prepare_ds(
        zarr_dir / "affs",
        shape=(3, 100, 100, 100),
        voxel_size=Coordinate(1, 1, 1),
        offset=Coordinate(0, 0, 0),
        dtype=np.float32,
        mode="w",
    )
    prepare_ds(
        zarr_dir / "frags",
        shape=(100, 100, 100),
        voxel_size=Coordinate(1, 1, 1),
        offset=Coordinate(0, 0, 0),
        dtype=np.uint64,
        mode="w",
    )
    prepare_ds(
        zarr_dir / "segments",
        shape=(100, 100, 100),
        voxel_size=Coordinate(1, 1, 1),
        offset=Coordinate(0, 0, 0),
        dtype=np.uint64,
        mode="w",
    )

    neighborhood = [Coordinate(0, 0, 1), Coordinate(0, 1, 0), Coordinate(1, 0, 0)]

    raw = Raw(store=zarr_dir / "raw")
    affs = Affs(store=zarr_dir / "affs", neighborhood=neighborhood)
    frags = Labels(store=zarr_dir / "frags")
    segments = Labels(store=zarr_dir / "segments")

    (tmpdir / "model").mkdir()
    model = torch.nn.Conv3d(1, 3, 1)
    torch.save(model, tmpdir / "model/model.pt")
    meta = {
        "num_in_channels": 1,
        "num_out_channels": 2,
        "neighborhood": neighborhood,
        "input_shape": [10, 10, 10],
        "output_shape": [10, 10, 10],
    }
    torch.save({"model_state_dict": model.state_dict()}, tmpdir / "model/checkpoint.pt")
    with open(tmpdir / "model/meta.json", "w") as f:
        f.write(json.dumps(meta))

    with open(tmpdir / "test.nml", "w") as f:
        f.write(
            textwrap.dedent("""<?xml version="1.0" encoding="UTF-8"?>
            <things>
                <thing id="1">
                    <nodes>
                        <node id="1" radius="1.5" x="4" y="5" z="5" inVp="0" inMag="1" time="0" comment="First Node"/>
                        <node id="2" radius="1.5" x="6" y="5" z="5" inVp="0" inMag="1" time="0"/>
                    </nodes>
                    <edges>
                        <edge source="1" target="2"/>
                    </edges>
                </thing>
            </things>""")
        )

    (zarr_dir / "luts").mkdir()
    np.savez(
        zarr_dir / "luts/test.npz",
        fragment_segment_lut=np.array([[0, 0], [1, 1]], dtype=np.uint64),
    )

    db_config = SQLite(
        path=tmpdir / "db.sqlite",
        node_attrs={"raw_intensity": 1},
        edge_attrs={
            "y_aff": "float",
            "z_aff": "float",
            "x_aff": "float",
            "raw_intensity_similarity": "float",
        },
    )
    db_config.open("w")

    pred_config = Predict(
        in_data=raw,
        out_data=[affs],
        checkpoint=Checkpoint(
            saved_model=tmpdir / "model/model.pt",
            meta_file=tmpdir / "model/meta.json",
            checkpoint_file=tmpdir / "model/checkpoint.pt",
        ),
    )
    extract_frags_config = ExtractFrags(
        db=db_config,
        affs_data=affs,
        frags_data=frags,
        block_size=Coordinate(10, 10, 10),
        context=Coordinate(0, 0, 0),
        save_intensities={"raw_intensity": raw},
        bias=[0.0, 0.0, 0.0],
    )
    seeded_extract_frags_config = SeededExtractFrags(
        db=db_config,
        affs_data=affs,
        segs_data=segments,
        block_size=Coordinate(10, 10, 10),
        context=Coordinate(0, 0, 0),
        bias=[0.0, 0.0, 0.0],
        nml_file=tmpdir / "test.nml",
    )
    aff_agglom_config = AffAgglom(
        db=db_config,
        frags_data=frags,
        affs_data=affs,
        block_size=Coordinate(10, 10, 10),
        context=Coordinate(0, 0, 0),
        scores={
            "z_aff": [Coordinate(1, 0, 0)],
            "y_aff": [Coordinate(0, 1, 0)],
            "x_aff": [Coordinate(0, 0, 1)],
        },
    )
    distance_agglom_config = DistanceAgglom(
        db=db_config,
        frags_data=frags,
        block_size=Coordinate(10, 10, 10),
        context=Coordinate(0, 0, 0),
        distance_keys=["raw_intensity"],
        distance_threshold=2,
    )
    global_mws_config = GlobalMWS(
        frags_data=frags,
        db=db_config,
        lut=zarr_dir / "luts" / "test.npz",
        bias={"x_aff": 0.0},
    )
    lut_config = LUT(
        frags_data=frags,
        seg_data=segments,
        lut=zarr_dir / "luts" / "test.npz",
        block_size=Coordinate(10, 10, 10),
    )
    argmax_config = Argmax(
        probs_data=raw,
        sem_data=segments,
        block_size=Coordinate(10, 10, 10),
    )

    return {
        "predict": pred_config,
        "extract_frags": extract_frags_config,
        "seeded_extract_frags": seeded_extract_frags_config,
        "aff_agglom": aff_agglom_config,
        "distance_agglom": distance_agglom_config,
        "global_mws": global_mws_config,
        "lut": lut_config,
        "argmax": argmax_config,
    }


@pytest.fixture()
def blockwise_configs(tmpdir):
    return build_configs(tmpdir)


@pytest.mark.parametrize(
    "task",
    [
        "predict",
        "extract_frags",
        "seeded_extract_frags",
        "aff_agglom",
        "distance_agglom",
        "global_mws",
        "lut",
        "argmax",
    ],
)
def test_block_funcs(blockwise_configs, task):
    config = blockwise_configs[task]
    assert config.meta_dir
    assert config.write_roi
    assert config.write_size
    assert config.process_block_func
    assert config.task()
    assert config.block_ds
    assert config.init

    with config.process_block_func() as process_block:
        block = daisy.Block(
            total_roi=Roi((0, 0, 0), (10, 10, 10)),
            read_roi=Roi((0, 0, 0), (10, 10, 10)),
            write_roi=Roi((0, 0, 0), (10, 10, 10)),
        )
        process_block(block)


def test_predict_block(blockwise_configs):
    config = blockwise_configs["predict"]
    block = daisy.Block(
        total_roi=Roi((0, 0, 0), (10, 10, 10)),
        read_roi=Roi((0, 0, 0), (10, 10, 10)),
        write_roi=Roi((0, 0, 0), (10, 10, 10)),
    )
    with config.process_block_func() as process_block:
        process_block(block)

    model = config.checkpoint_config.model()
    raw_data_in = config.in_data.array("r").to_ndarray(block.read_roi)
    affs_out = model(torch.tensor(raw_data_in[None, ...])).detach().numpy()

    written_data = config.out_data[0].array("r").to_ndarray(block.write_roi)
    assert np.allclose(written_data, np.clip(affs_out * 255, 0, 255).astype(np.uint8))


def test_extract_frags_block(blockwise_configs):
    config = blockwise_configs["extract_frags"]
    block = daisy.Block(
        total_roi=Roi((0, 0, 0), (10, 10, 10)),
        read_roi=Roi((0, 0, 0), (10, 10, 10)),
        write_roi=Roi((0, 0, 0), (10, 10, 10)),
    )
    affs_array = config.affs_data.array("r+")
    block_affs = np.ones((3, 10, 10, 10), dtype=np.float32) - 2 * np.array(
        config.bias
    ).reshape(-1, 1, 1, 1)
    block_affs[0, :, :, 5] *= -1
    affs_array[block.write_roi] = block_affs
    with config.process_block_func() as process_block:
        process_block(block)
    frags_out = config.frags_data.array("r").to_ndarray(block.read_roi)
    assert 0 not in np.unique(frags_out)
    assert np.unique(frags_out[:, :, :6]).size == 1
    assert np.unique(frags_out[:, :, 6:]).size == 1
    assert np.unique(frags_out).size == 2

    db = config.db.open("r")
    graph = db.read_graph(block.write_roi)
    assert len(graph.nodes) == 2


def test_seeded_extract_frags_block(blockwise_configs):
    config = blockwise_configs["seeded_extract_frags"]
    block = daisy.Block(
        total_roi=Roi((0, 0, 0), (10, 10, 10)),
        read_roi=Roi((0, 0, 0), (10, 10, 10)),
        write_roi=Roi((0, 0, 0), (10, 10, 10)),
    )
    affs_array = config.affs_data.array("r+")
    block_affs = np.ones((3, 10, 10, 10), dtype=np.float32) - 2 * np.array(
        config.bias
    ).reshape(-1, 1, 1, 1)
    block_affs[0, :, :, 5] *= -1
    affs_array[block.write_roi] = block_affs
    with config.process_block_func() as process_block:
        process_block(block)
    frags_out = config.segs_data.array("r").to_ndarray(block.read_roi)
    assert 0 not in np.unique(frags_out)
    assert np.unique(frags_out[:, :, :6]).size == 1
    assert np.unique(frags_out[:, :, 6:]).size == 1
    assert np.unique(frags_out).size == 1


def test_aff_agglom_block(blockwise_configs):
    config: AffAgglom = blockwise_configs["aff_agglom"]
    block = daisy.Block(
        total_roi=Roi((0, 0, 0), (10, 10, 10)),
        read_roi=Roi((0, 0, 0), (10, 10, 10)),
        write_roi=Roi((0, 0, 0), (10, 10, 10)),
    )
    affs_array = config.affs_data.array("r+")
    block_affs = np.ones((3, 10, 10, 10), dtype=np.float32)
    block_affs[0, :, :, 5] = 0.07
    affs_array[block.write_roi] = block_affs

    frags_array = config.frags_data.array("r+")
    block_frags = np.ones((10, 10, 10), dtype=np.uint64)
    block_frags[:, :, 6:] = 2
    frags_array[block.write_roi] = block_frags

    db = config.db.open("r+")
    graph = db.read_graph(block.write_roi)
    graph.add_node(1, position=(4, 4, 2), size=600, raw_intensity=(0.1,))
    graph.add_node(2, position=(4, 4, 7), size=400, raw_intensity=(0.1,))
    db.write_graph(graph, roi=block.write_roi)

    with config.process_block_func() as process_block:
        process_block(block)

    graph = db.read_graph(block.write_roi)
    assert graph.number_of_edges() == 1
    assert np.isclose(graph.edges[(1, 2)]["x_aff"], 0.07)
    assert graph.edges[(1, 2)]["z_aff"] is None
    assert graph.edges[(1, 2)]["y_aff"] is None
    with pytest.raises(KeyError):
        graph.edges[(1, 2)]["xyz_aff"]


@pytest.mark.parametrize(
    "distance_metric_and_score", [("cosine", -1.0), ("euclidean", -0.2)]
)
def test_distance_agglom_block(blockwise_configs, distance_metric_and_score):
    metric, raw_score = distance_metric_and_score
    config: DistanceAgglom = blockwise_configs["distance_agglom"]
    config.distance_metric = metric
    block = daisy.Block(
        total_roi=Roi((0, 0, 0), (10, 10, 10)),
        read_roi=Roi((0, 0, 0), (10, 10, 10)),
        write_roi=Roi((0, 0, 0), (10, 10, 10)),
    )
    frags_array = config.frags_data.array("r+")
    block_frags = np.ones((10, 10, 10), dtype=np.uint64)
    block_frags[:, :, 6:] = 2
    frags_array[block.write_roi] = block_frags

    db = config.db.open("r+")
    graph = db.read_graph(block.write_roi)
    graph.add_node(1, position=(4, 4, 2), size=600, raw_intensity=(0.1,))
    graph.add_node(
        2,
        position=(4, 4, 7),
        size=400,
        raw_intensity=(-0.1,),
    )
    db.write_graph(graph, roi=block.write_roi)

    with config.process_block_func() as process_block:
        process_block(block)

    graph = db.read_graph(block.write_roi)
    assert graph.number_of_edges() == 1
    assert np.isclose(
        graph.edges[(1, 2)]["raw_intensity_similarity"], raw_score
    ), graph.edges[(1, 2)]


@pytest.mark.parametrize("x_aff", [0.5, -0.5])
def test_global_mws_block(blockwise_configs, x_aff):
    config: GlobalMWS = blockwise_configs["global_mws"]
    block = daisy.Block(
        total_roi=Roi((0, 0, 0), (10, 10, 10)),
        read_roi=Roi((0, 0, 0), (10, 10, 10)),
        write_roi=Roi((0, 0, 0), (10, 10, 10)),
    )

    db = config.db.open("r+")
    graph = db.read_graph(block.write_roi)
    graph.add_node(1, position=(4, 4, 2), size=600, raw_intensity=(0.1,))
    graph.add_node(2, position=(4, 4, 7), size=400, raw_intensity=(0.1,))
    graph.add_edge(
        1,
        2,
        x_aff=x_aff,
    )
    db.write_graph(graph, roi=block.write_roi)

    with config.process_block_func() as process_block:
        process_block(block)

    lut = np.load(config.lut)["fragment_segment_lut"]
    assert len(np.unique(lut[0, :])) == 2, lut
    assert len(np.unique(lut[1, :])) == 1 + (x_aff < 0), lut


def test_lut_block(blockwise_configs):
    config: LUT = blockwise_configs["lut"]
    block = daisy.Block(
        total_roi=Roi((0, 0, 0), (10, 10, 10)),
        read_roi=Roi((0, 0, 0), (10, 10, 10)),
        write_roi=Roi((0, 0, 0), (10, 10, 10)),
    )

    np.savez(config.lut, fragment_segment_lut=np.array([[1, 2], [100, 200]]))

    frags_array = config.frags_data.array("r+")
    block_frags = np.ones((10, 10, 10), dtype=np.uint64)
    block_frags[:, :, 6:] = 2
    frags_array[block.write_roi] = block_frags

    with config.process_block_func() as process_block:
        process_block(block)

    segs_array = config.seg_data.array("r").to_ndarray(block.write_roi)
    assert (segs_array[:, :, :6].min(), segs_array[:, :, :6].max()) == (100, 100)
    assert (segs_array[:, :, 6:].min(), segs_array[:, :, 6:].max()) == (200, 200)
