import pytest
import torch

from src.trainers.faceShifter_trainer import FaceShifterTrainer


@pytest.mark.skip
class TestFaceShifterTrainer:
    @pytest.fixture
    def opt(self):
        return {
            "FaceShifterTrainer":{
                "max_epochs": 1,
                "continue":{
                    "status": False,
                    "epoch": 0,
                },
                "d_step_per_g": 1,
                "optim":{
                    "name": "Adam",
                    "lr": 1e-3,
                    "betas": [0.1, 0.2],  
                },            
            },
            "AEINetLoss":{
                "weights": {
                    "AttrLoss": 10,
                    "RecLoss": 10,
                    "IdtLoss": 5
                }
            },
            "MultiScaleDiscriminator": {
                "num_ds": 3,
                "downsample_scale_factor": 0.5,        
                "PatchGANDiscriminator": {
                    "num_conv_blks": 6,
                    "in_channels": 3,
                    "LeakyReLU_slope": 0.2,
                    "conv":{
                        "out_channel_list": [64, 128, 256, 512, 512, 512],
                        "kernel_size": 4,
                        "stride": 2,
                        "padding": 2 , 
                    }  
                } 
            },
            "IdtEncoder":{
                "name": "ArcFace",
                "pretrained_model": "experiments/idt_encoder/ArcFace.pth"           
            }
        }


    # @pytest.mark.skip
    def test_fit_g(self, opt):
        trainer = FaceShifterTrainer(opt)
        xs = torch.rand((1,3,256,256))
        xt = torch.rand((1,3,256,256))

        trainer.fit_g(xs, xt)


    # @pytest.mark.skip
    def test_fit_d(self, opt):
        trainer = FaceShifterTrainer(opt)
        xs = torch.rand((1,3,256,256))
        xt = torch.rand((1,3,256,256))

        trainer.fit_d(xs, xt)



