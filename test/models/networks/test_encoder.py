import torch
import pytest


from src.models.networks.encoder import AttrEncoder, IdtEncoder


@pytest.mark.skip
class TestAttrEncoder:
    def test_forward_out_size(self):
        ae = AttrEncoder()
        data = torch.rand((1, 3, 256, 256))
        result = ae(data)
        expected_size = (1, 64, 256, 256)
        assert result.size() == expected_size


    
    def test_decoder_feature_maps(self):
        """Test the number and sizes of feature maps of the decoders
        """
        ae = AttrEncoder()
        data = torch.rand((1, 3, 256, 256))
        _ = ae(data)
        decoder_features = ae.get_decoder_features()
        expected_size = [
            (1,1024,2,2),
            (1,2048,4,4),
            (1,1024,8,8),
            (1,512,16,16),
            (1,256,32,32),
            (1,128,64,64),
            (1,64,128,128),
            (1,64,256,256)
        ]
        for feature, s in zip(decoder_features, expected_size):
            assert feature.size() == s


@pytest.mark.skip
class TestIdtEncoder:
    def test_forward_out_size(self):
        x = torch.rand((1, 3, 256, 256))
        idt = IdtEncoder()
        output = idt(x)

        expected_size = (1, 256, 1, 1)

        assert output.size() == expected_size