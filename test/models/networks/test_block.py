import pytest
import torch

from src.models.networks.block import AADResBlk


@pytest.mark.skip
class TestAADResBlk:
    # @pytest.mark.skip    
    def test_forward_out_size_1(self):
        """Test if forward returns an output with expected size, when the number
        of input channels is different from that of output channels
        """
        blk = AADResBlk(32, 16, 256, 8)
        h = torch.rand((2, 32, 10, 10))
        idt = torch.rand((2, 256, 1, 1))
        attr = torch.rand((2, 16, 10, 10))

        out = blk((h, idt, attr))
        expected_size = (2, 8, 10, 10)

        assert out.size() == expected_size


    # @pytest.mark.skip
    def test_forward_out_size_2(self):
        """Test if forward returns an output with expected size, when the number
        of input channels is the same as that of output channels  
        """
        blk = AADResBlk(32, 16, 256, 32)
        h = torch.rand((2, 32, 10, 10))
        idt = torch.rand((2, 256, 1, 1))
        attr = torch.rand((2, 16, 10, 10))

        out = blk((h, idt, attr))
        expected_size = (2, 32, 10, 10)

        assert out.size() == expected_size