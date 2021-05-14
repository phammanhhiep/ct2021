import torch
import pytest


from src.models.networks.generator import AEINet, AADGenerator


@pytest.mark.skip
class TestAADGenerator:
    def test_forward_out_size(self):
        g = AADGenerator()
        idt = torch.rand((1, 256, 1, 1))
        attr = [
            torch.rand((1, 1024, 2, 2)),
            torch.rand((1, 2048, 4, 4)),
            torch.rand((1, 1024, 8, 8)),
            torch.rand((1, 512, 16, 16)),
            torch.rand((1, 256, 32, 32)),
            torch.rand((1, 128, 64, 64)),
            torch.rand((1, 64, 128, 128)),
            torch.rand((1, 64, 256, 256)),
            ]
        result = g(idt, attr)
        expected_size = (1, 3, 256, 256)
        assert result.size() == expected_size


@pytest.mark.skip
class TestAEINet:
    def test_forward_out_size(self):
        xs = torch.rand((1,3,256,256))
        xt = torch.rand((1,3,256,256))
        aei = AEINet()
        result = aei(xs, xt)

        expected_size = (1, 3, 256, 256)
        assert result.size() == expected_size
