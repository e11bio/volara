import logging
from contextlib import contextmanager
from shutil import rmtree
from typing import Annotated, Callable, Literal

import daisy
import gunpowder as gp
import numpy as np
import torch
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array
from gunpowder import ArrayKey, Batch, BatchProvider
from pydantic import Field

from ..datasets import LSD, Affs, Dataset, Raw
from ..models import Model, TorchModel
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask

logger = logging.getLogger(__file__)


class ArraySource(BatchProvider):
    def __init__(self, key: ArrayKey, array: Array):
        self.key = key
        self.array = array

    def setup(self):
        spec = gp.ArraySpec(self.array.roi, self.array.voxel_size)
        self.provides(self.key, spec)

    def provide(self, request):
        outputs = Batch()
        data = self.array[request[self.key].roi]
        outputs[self.key] = gp.Array(
            data=data,
            spec=gp.ArraySpec(
                roi=request[self.key].roi,
                voxel_size=self.array.voxel_size,
            ),
        )
        return outputs


class ArrayWrite(gp.BatchFilter):
    def __init__(self, key: ArrayKey, array: Array, to_uint8: Callable):
        self.key = key
        self.array = array
        self.to_uint8 = to_uint8

    def setup(self):
        self.updates(self.key, self.spec[self.key].copy())

    def process(self, batch, request):
        write_roi = request[self.key].roi.intersect(self.array.roi)
        data = batch[self.key].crop(write_roi).data
        self.array[write_roi] = self.to_uint8(data)


OutDataType = Annotated[
    Raw | Affs | LSD,
    Field(discriminator="dataset_type"),
]


class Predict(BlockwiseTask):
    task_type: Literal["predict"] = "predict"
    roi: tuple[PydanticCoordinate, PydanticCoordinate] | None = None
    checkpoint: Annotated[
        TorchModel,
        Field(discriminator="checkpoint_type"),
    ]
    in_data: Raw
    out_data: list[OutDataType | None]

    fit: Literal["overhang"] = "overhang"
    read_write_conflict: Literal[False] = False
    _out_array_dtype: np.dtype = np.dtype(np.uint8)

    @property
    def checkpoint_config(self) -> Model:
        return self.checkpoint

    @property
    def write_roi(self) -> Roi:
        in_data_roi = self.in_data.array("r").roi
        if self.roi is not None:
            return in_data_roi.intersect(Roi(self.roi[0], self.roi[1]))
        else:
            return in_data_roi

    @property
    def voxel_size(self) -> Coordinate:
        return self.in_data.array("r").voxel_size

    @property
    def write_size(self) -> Coordinate:
        return self.checkpoint_config.eval_output_shape * self.voxel_size

    @property
    def context_size(self) -> Coordinate:
        return self.checkpoint_config.context * self.voxel_size

    @property
    def task_name(self) -> str:
        return f"{'-'.join(dataset.name for dataset in self.output_datasets)}-{self.task_type}"

    @property
    def output_datasets(self) -> list[Dataset]:
        return [out_data for out_data in self.out_data if out_data is not None]

    def drop_artifacts(self):
        for out_data in self.out_data:
            if out_data is not None:
                rmtree(out_data.store)

    def init(self):
        self.init_out_array()

    def init_out_array(self):
        # TODO: hardcoding assuming channels first :/
        in_data = self.in_data.array("r")
        units = in_data.units
        axis_names = in_data.axis_names[-in_data.voxel_size.dims :]
        for out_data, num_channels in zip(
            self.out_data, self.checkpoint_config.num_out_channels
        ):
            if num_channels is None:
                num_channels = self.in_data.array("r").shape[0]
            if out_data is not None:
                shape = (num_channels, *(self.write_roi.shape / self.voxel_size))
                chunk_shape = (num_channels, *self.write_size / self.voxel_size)
                out_data.prepare(
                    shape=shape,
                    chunk_shape=chunk_shape,
                    offset=self.write_roi.offset,
                    voxel_size=self.voxel_size,
                    units=units,
                    axis_names=[f"{out_data.name}^"] + axis_names,
                    types=[out_data.name] + in_data.types,
                    dtype=self._out_array_dtype,
                    kwargs=out_data.attrs,
                )

    def select_device(self, client):
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            return "cpu"
        else:
            device_id = client.worker_id % num_gpus
            return f"cuda:{device_id}"

    @contextmanager
    def process_block_func(self):
        try:
            client = daisy.Client()
            device = self.select_device(client)
        except KeyError:
            device = "cuda"

        logging.info(f"using device {device}")

        input_key = gp.ArrayKey("INPUT_KEY")
        output_keys = [
            gp.ArrayKey(f"OUTPUT_KEY_{i}") for i in range(len(self.out_data))
        ]

        model = self.checkpoint.model()
        model.eval()

        in_array = self.in_data.array("r")

        pipeline = ArraySource(input_key, in_array)

        pipeline += gp.Pad(input_key, size=None)

        if in_array.channel_dims < 2:
            pipeline += gp.Stack(1)
        if in_array.channel_dims < 1:
            pipeline += gp.Stack(1)

        pipeline += gp.torch.Predict(
            model=model,
            inputs={0: input_key},
            outputs={
                i: output_key
                for i, output_key in enumerate(output_keys)
                if self.out_data[i] is not None
            },
            device=device,
            spawn_subprocess=self.num_cache_workers is not None
            and self.num_cache_workers > 1,
        )

        pipeline += gp.Squeeze(
            [
                output_key
                for i, output_key in enumerate(output_keys)
                if self.out_data[i] is not None
            ]
        )

        for output_key, out_data in zip(output_keys, self.out_data):
            if out_data is not None:
                pipeline += ArrayWrite(
                    output_key, out_data.array("a"), self.checkpoint.to_uint8
                )

        print("Starting prediction...")

        with gp.build(pipeline):

            def process_block(block):
                request = gp.BatchRequest()
                request[input_key] = gp.ArraySpec(roi=block.read_roi)
                for i, output_key in enumerate(output_keys):
                    if self.out_data[i] is not None:
                        request[output_key] = gp.ArraySpec(roi=block.write_roi)
                pipeline.request_batch(request)

            yield process_block
