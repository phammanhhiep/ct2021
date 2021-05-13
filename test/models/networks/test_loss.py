import pytest
import torch


from src.models.networks.loss import AEINetLoss, MultiScaleGanLoss, AttrLoss, \
    RecLoss, IdtLoss


@pytest.mark.skip
class TestMultiScaleGanLoss:
    def test_forward_out_size(self):
        criterion = MultiScaleGanLoss()
        x = [torch.rand((1,1,3,3)) for i in  range(3)]
        loss = criterion(x, label=True, compute_d_loss=True)

        assert len(loss.size()) == 0


    def test_loss_value(self):
        criterion = MultiScaleGanLoss()
        x = [torch.rand((1,1,3,3)) for i in  range(3)]
        loss = criterion(x, label=False, compute_d_loss=True)

        assert loss != 0


@pytest.mark.skip
class TestAttrLoss:
    def test_forward_out_size(self):
        criteron = AttrLoss()
        x = [torch.rand((1,2,3,3)) for i in range(3)]
        y = [torch.rand((1,2,3,3)) for i in range(3)]
        loss = criteron(x, y)

        assert len(loss.size()) == 0


@pytest.mark.skip
class TestRecLoss:
    def test_forward_out_size(self):
        criteron = RecLoss()
        x = torch.rand((1,3,128,128))

        loss = criteron(x, x, reconstructed=True)
        assert len(loss.size()) == 0


@pytest.mark.skip
class TestIdtLoss:
    def test_forward_out_size(self):
        criteron = IdtLoss()
        x = torch.rand((1,256,1,1))
        y = torch.rand((1,256,1,1))

        loss = criteron(x, y)

        assert len(loss.size()) == 0


@pytest.mark.skip
class TestAEINetLoss:
    def test_forward_out_size(self):
        criteron = AEINetLoss({
            "weights": {
                "AttrLoss": 1,
                "RecLoss": 1,
                "IdtLoss": 1
            }
            })
        xt =  torch.rand((1,3,128,128))
        y = torch.rand((1,3,128,128)) 
        xt_attr = [torch.rand((1,2,3,3)) for i in range(3)]
        y_attr = [torch.rand((1,2,3,3)) for i in range(3)]
        xs_idt = torch.rand((1,256,1,1))
        y_idt = torch.rand((1,256,1,1))
        d_output = [torch.rand((1,1,3,3)) for i in  range(3)]

        loss = criteron(xt, y, xt_attr, y_attr, xs_idt, y_idt, d_output, 
            reconstructed=False)

        assert len(loss.size()) == 0