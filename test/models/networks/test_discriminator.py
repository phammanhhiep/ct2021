import pytest
import torch


from src.models.networks.discriminator import MultiScaleDiscriminator, \
    PatchGANDiscriminator


@pytest.mark.skip
class TestPatchGANDiscriminator:
    def test_forward_out_size(self):
        d = PatchGANDiscriminator()
        x = torch.rand((1, 3, 256, 256))
        result = d(x)
        expected_size = (1,1,3,3)
        assert result.size() == expected_size


@pytest.mark.skip
class TestMultiScaleDiscriminator:
    def test_forward_out_size(self):
        m = MultiScaleDiscriminator()

        x = torch.rand((1, 3, 256, 256))
        result = m(x)
        
        expected_size = [
            (1,1,3,3),
            (1,1,2,2),
            (1,1,2,2)
        ]
        assert len(result) == 3
        for p, s in zip(result, expected_size):
            assert p.size() == s