import pytest
import torch


from src.models.networks.normalization import AADNorm


@pytest.mark.skip
class TestAADNorm:
    def test_forward_out_size(self):
        """Test if forward returns an output with expected size 
        """
        norm = AADNorm(16, 32, 256)
        h = torch.rand((2, 16, 10, 10))
        idt = torch.rand((2, 256, 1, 1))
        attr = torch.rand((2, 32, 10, 10))

        out = norm((h, idt, attr))

        assert out.size() == h.size()