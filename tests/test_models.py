import numpy as np
import pytest
import torch

from volara.models import TorchModel


@pytest.mark.parametrize("save_type", ["jit", "pickle"])
@pytest.mark.parametrize("out_range", [(0, 1), (-1, 1)])
@pytest.mark.parametrize("checkpoint", [True, False])
def test_torch_models(save_type, out_range, checkpoint, tmp_path):
    test_data = torch.randn(1, 1, 100, 100)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 1, 3, padding="same"), torch.nn.Sigmoid()
    )
    test_output = model(test_data)
    weights = model.state_dict()
    checkpoint_file = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": weights}, checkpoint_file)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 1, 3, padding="same"), torch.nn.Sigmoid()
    )
    test_output2 = model(test_data)
    assert not torch.allclose(test_output, test_output2)
    if save_type == "jit":
        save_path = tmp_path / "model.jit"
        trace = torch.jit.trace(model, test_data)
        torch.jit.save(trace, save_path)
    elif save_type == "pickle":
        save_path = tmp_path / "model.pth"
        torch.save(model, save_path)

    model_config = TorchModel(
        in_channels=1,
        min_input_shape=(1, 1),
        min_output_shape=(1, 1),
        min_step_shape=(1, 1),
        out_channels=1,
        out_range=out_range,
        save_path=save_path,
        checkpoint_file=checkpoint_file if checkpoint else None,
        pred_size_growth=(99, 99),
    )

    # basic model interface values
    assert model_config.context == (0, 0)
    assert model_config.eval_input_shape == (100, 100)
    assert model_config.eval_output_shape == (100, 100)
    assert model_config.num_out_channels == [1]

    # model data conversions
    uint8_data = np.arange(0, 256, dtype=np.uint8).reshape(16, 16)
    float_data = model_config.from_uint8(uint8_data)
    uint8_data_back = model_config.to_uint8(float_data)
    assert np.isclose(float_data.max(), model_config.out_range[1])
    assert np.isclose(float_data.min(), model_config.out_range[0])
    assert uint8_data_back.max() == 255, (
        uint8_data.min(),
        uint8_data.max(),
        float_data.min(),
        float_data.max(),
        uint8_data_back.min(),
        uint8_data_back.max(),
    )
    assert uint8_data_back.min() == 0

    # test model runs as expected
    if checkpoint:
        assert torch.allclose(model_config.model()(test_data), test_output)
    else:
        assert torch.allclose(model_config.model()(test_data), test_output2)
