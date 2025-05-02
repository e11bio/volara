from contextlib import contextmanager
from typing import Literal
from pathlib import Path

import numpy as np
import os
import logging

from funlib.geometry import Coordinate, Roi
from volara.blockwise import BlockwiseTask
from volara.datasets import Dataset, CloudVolumeWrapper
from volara.utils import PydanticCoordinate
from daisy import Block


class SamplePointCloud(BlockwiseTask):
    task_type: Literal["sample_pc"] = "sample_pc"
    out_dir: str
    labels: CloudVolumeWrapper
    block_size: PydanticCoordinate
    fraction: float
    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False
    sample_svids: bool = True
    

    @property
    def task_name(self) -> str:
        return f"{self.labels.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        total_roi = self.labels.roi
        if self.roi is not None:
            total_roi = total_roi.intersect(Roi(self.roi[0], self.roi[1]))
        return total_roi


    @property
    def voxel_size(self) -> Coordinate:
        return self.labels.voxel_size

    @property
    def write_size(self) -> Coordinate:
        return self.block_size

    @property
    def context_size(self) -> Coordinate:
        return Coordinate((0,) * self.write_size.dims)

    def drop_artifacts(self):
        pass

    @property
    def output_datasets(self) -> list[Dataset]:
        return []
    
    def sample_pc_in_block(self, block: Block, labels: CloudVolumeWrapper):
        block_id = block.block_id[1]
        data = labels.data

        labels_data = data[block.write_roi.to_slices()] # TODO: check if XYZ vs ZYX is correct
        
        # make labels data a numpy array
        labels_data = np.array(labels_data).squeeze()

        logging.info(f"got {len(np.unique(labels_data))} in {block_id}")

        sampled_points = self.sample_segment_points(labels_data, self.fraction)

        if self.sample_svids:
            data.agglomerate = False
            svid_data = data[block.write_roi.to_slices()]
            svid_data = np.array(svid_data).squeeze()
            data.agglomerate = True

        # add block offset to points
        offset = block.write_roi.get_begin()
        for seg, pts in sampled_points.items():
            packed_pts = pts + offset
            if self.sample_svids:
                x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
                svids = svid_data[x, y, z].reshape(-1, 1)
                packed_pts = np.hstack((packed_pts, svids)).astype(np.int64)
            sampled_points[seg] = packed_pts

        # if writing all labels to blocks rather than per label
        # out_f = os.path.join(out_dir, f"block_{block_id}.npz")
        # np.savez_compressed(out_f, **sampled_points)

        root = Path(self.out_dir)
        for seg, pts in sampled_points.items():
            label_dir = root / f"label_{seg}"
            label_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(label_dir / f"block_{block_id}.npz", points=pts)

    def init(self):
        os.makedirs(self.out_dir, exist_ok=True)

    @contextmanager
    def process_block_func(self):

        def process_block(block: Block):
            self.sample_pc_in_block(block, self.labels)

        yield process_block


    def sample_segment_points(self, labels, fraction, background=0, replace=False):
        segment_ids = np.unique(labels)
        if background is not None:
            segment_ids = segment_ids[segment_ids != background]

        segment_points = {}

        for seg in segment_ids:
            coords = np.column_stack(np.where(labels == seg))
            num_points = coords.shape[0]

            k = max(1, int(num_points * fraction))

            if num_points > k:
                selected_idx = np.random.choice(num_points, size=k, replace=replace)
                sampled = coords[selected_idx]
            else:
                sampled = coords

            segment_points[str(seg)] = sampled

        return segment_points