import pytest
import torch


from src.models.faceShifter_model import FaceShifterModel


@pytest.mark.skip
class TestFaceShifterModel:
    def test_forward_out_size(self):
        m = FaceShifterModel()
        xs = torch.rand((1,3,256,256))
        xt = torch.rand((1,3,256,256))

        result = m(xs, xt, mode=2)
        assert len(result) == 2

        result = m(xs, xt, mode=1)
        assert result.size() == (1,3,256, 256)
