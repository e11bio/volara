import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from funlib.geometry import Coordinate

from .utils import PydanticCoordinate, StrictBaseModel


class Model(ABC, StrictBaseModel):
    """
    A base class for defining the common attributes and methods for all
    model types.
    """
    @property
    def context(self):
        return (self.eval_input_shape - self.eval_output_shape) // 2

    @abstractmethod
    def model(self):
        pass

    @property
    @abstractmethod
    def eval_input_shape(self) -> Coordinate:
        pass

    @property
    @abstractmethod
    def eval_output_shape(self) -> Coordinate:
        pass

    @property
    @abstractmethod
    def num_out_channels(self) -> list[int]:
        pass

    def to_uint8(self, out_data):
        return np.clip(out_data * 255, 0, 255).astype(np.uint8)

    def from_uint8(self, data):
        return data.astype(np.float32) / 255


class Checkpoint(Model):
    checkpoint_type: Literal["checkpoint"] = "checkpoint"
    saved_model: Path  # tmp
    checkpoint_file: Path
    meta_file: Path
    pred_size_growth: Optional[PydanticCoordinate] = None

    def model(self):
        import torch

        model = torch.load(self.saved_model, map_location="cpu")

        model.load_state_dict(
            torch.load(self.checkpoint_file, map_location="cpu")["model_state_dict"]
        )

        if "loss_func" in self.conf and self.conf["loss_func"] == "classification":
            model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))

        try:
            model.num_in_channels = self.conf["num_in_channels"]
        except KeyError:
            model.num_in_channels = self.conf["in_channels"]

        if "neighborhood" in self.conf:
            model.neighborhood = self.conf["neighborhood"]

        return model

    @property
    def conf(self) -> dict:
        with open(self.meta_file) as f:
            meta_data = json.load(f)
            if isinstance(meta_data, str):
                meta_data = json.loads(meta_data)
            if "m_conf" in meta_data:
                meta_data = meta_data["m_conf"]
        return meta_data

    @property
    def eval_input_shape(self) -> Coordinate:
        input_shape = Coordinate(self.conf["input_shape"])
        if self.pred_size_growth is not None:
            input_shape = input_shape + self.pred_size_growth
        return input_shape

    @property
    def eval_output_shape(self) -> Coordinate:
        output_shape = Coordinate(self.conf["output_shape"])
        if self.pred_size_growth is not None:
            output_shape = output_shape + self.pred_size_growth
        return output_shape

    @property
    def num_out_channels(self) -> list[int]:
        try:
            num_channels = self.conf["num_out_channels"]
        except KeyError:
            num_channels = self.conf["num_fmaps_out"]

        if isinstance(num_channels, int):
            return [num_channels]
        elif isinstance(num_channels, list):
            return num_channels
        else:
            raise ValueError(
                f"Output channels ({num_channels}) must be an int or a list of ints."
            )


class DaCapo(Model):
    checkpoint_type: Literal["dacapo"] = "dacapo"
    name: str
    criterion: str

    def model(self):
        from dacapo.experiments import Run
        from dacapo.store.create_store import create_config_store, create_weights_store

        config_store = create_config_store()
        weights_store = create_weights_store()

        run_config = config_store.retrieve_run_config(self.name)
        run = Run(run_config)
        model = run.model
        try:
            weights_store._load_best(run, self.criterion)
        except FileNotFoundError:
            iteration = int(self.criterion)
            weights = weights_store.retrieve_weights(run, iteration)
            model.load_state_dict(weights.model)

        eval_output_shape = model.compute_output_shape(model.eval_input_shape)[1]
        model.eval_output_shape = eval_output_shape

        return model

    @property
    def eval_input_shape(self) -> Coordinate:
        raise NotImplementedError()

    @property
    def eval_output_shape(self) -> Coordinate:
        raise NotImplementedError()

    @property
    def num_out_channels(self) -> list[int]:
        raise NotImplementedError()
