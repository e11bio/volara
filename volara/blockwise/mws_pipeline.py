import daisy
from funlib.geometry import Roi

from volara.lut import LUT

from ..datasets import Affs, Labels, Raw
from ..dbs import PostgreSQL, SQLite
from ..models import TorchModel
from ..utils import PydanticCoordinate
from ..workers import Worker
from .components import (
    AffAgglom,
    ExtractFrags,
    GraphMWS,
    Predict,
    Relabel,
)


class MWSPipeline:
    raw_config: Raw
    affs_config: Affs
    frags_config: Labels
    segs_config: Labels

    num_cache_workers: int = 1
    num_workers_per_task: int = 1
    worker_config: Worker | None = None

    block_size: PydanticCoordinate
    context: PydanticCoordinate

    bias: list[float]
    filter_fragments: float
    remove_debris: int
    lut: LUT
    edge_biases: dict[str, float] = {"adj_weight": -0.0, "lr_weight": -1.0}
    scores: dict[str, list[PydanticCoordinate]]

    affs_model_config: TorchModel | None = None

    db_config: PostgreSQL | SQLite

    roi: tuple[PydanticCoordinate, PydanticCoordinate]

    @property
    def write_roi(self) -> Roi:
        raise NotImplementedError()

    @property
    def write_size(self) -> PydanticCoordinate:
        raise NotImplementedError()

    def init(self) -> None:
        if self.affs_pred_config is not None:
            self.affs_pred_config.init()
        self.extract_frags_config.init()
        self.agglom_frags_config.init()
        self.global_segments_config.init()
        self.write_segments_config.init()

    @property
    def process_block_func(self):
        raise NotImplementedError()

    @property
    def affs_pred_config(self) -> Predict | None:
        if self.affs_model_config is None:
            return None
        else:
            return Predict(
                roi=self.roi,
                checkpoint=self.affs_model_config,
                in_data=self.raw_config,
                out_data=[self.affs_config],
                num_workers=self.num_workers_per_task,
                num_cache_workers=self.num_cache_workers,
                worker_config=self.worker_config,
            )

    @property
    def extract_frags_config(self) -> ExtractFrags:
        return ExtractFrags(
            db=self.db_config,
            affs_data=self.affs_config,
            frags_data=self.frags_config,
            block_size=self.block_size,
            context=self.context,
            num_workers=self.num_workers_per_task,
            worker_config=self.worker_config,
            bias=self.bias,
        )

    @property
    def agglom_frags_config(self) -> AffAgglom:
        return AffAgglom(
            db=self.db_config,
            frags_data=self.frags_config,
            affs_data=self.affs_config,
            block_size=self.block_size,
            context=self.context,
            num_workers=self.num_workers_per_task,
            worker_config=self.worker_config,
            scores=self.scores,
        )

    @property
    def global_segments_config(self) -> GraphMWS:
        return GraphMWS(
            db=self.db_config,
            lut=self.lut,
            weights={k: (1.0, v) for k, v in self.edge_biases.items()},
            num_workers=self.num_workers_per_task,
            roi=self.roi,
            worker_config=self.worker_config,
        )

    @property
    def write_segments_config(self) -> Relabel:
        return Relabel(
            frags_data=self.frags_config,
            seg_data=self.segs_config,
            lut=self.lut,
            block_size=self.block_size,
            num_workers=self.num_workers_per_task,
            worker_config=self.worker_config,
        )

    def task(
        self, upstream_tasks: daisy.Task | list[daisy.Task] | None = None
    ) -> daisy.Task:
        assert upstream_tasks is None, "MWSPipeline does not accept upstream tasks!"

        if self.affs_pred_config is None:
            affs_pred_task = None
        else:
            affs_pred_task = self.affs_pred_config.task()

        upstream_tasks = [affs_pred_task] if affs_pred_task is not None else None
        extract_frags_task = self.extract_frags_config.task(upstream_tasks)
        agglom_frags_task = self.agglom_frags_config.task(extract_frags_task)
        global_segments_task = self.global_segments_config.task(agglom_frags_task)
        write_segments_task = self.write_segments_config.task(global_segments_task)

        return write_segments_task

    def run_blockwise(self, multiprocessing: bool = False) -> None:
        self.init()
        task = self.task()
        if multiprocessing:
            daisy.run_blockwise(task)
        else:
            server = daisy.SerialServer()
            _cl_monitor = daisy.cl_monitor.CLMonitor(server)
            server.run_blockwise([task])
