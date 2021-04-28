from torch import nn


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.num_ds = opt["num_ds"]
        self.ds_scale_factors = opt["downsample_scale_factors"]
        self.ds = []

        for n in range(self.num_ds):
            self.ds.append(PatchGANDiscriminator(opt["patch_gan_d"]))


    def forward(self, x):
        y = []
        for n in range(self.num_ds):
            x = nn.functional.interpolate(x, 
                (1, 1, self.ds_scale_factors[n], self.ds_scale_factors[n]))
            y.append(self.ds[n](x))
        return y


class PatchGANDiscriminator(nn.Module):

    """Implementation of the Patch-GAN described in "[2017] Image-to-image 
    translation with conditional adversarial networks"
    """

    #TODO: verify if spectral norm is used at the same time with instance norm, or instance norm is replaced by spectral norm
    #TODO: include an option for spectral norms
    def __init__(self, opt):
        super().__init__()
        conv_blk_opt = opt["conv_blk"]
        in_channels = conv_blk_opt["conv"]["in_channels"]
        out_channels = conv_blk_opt["conv"]["out_channels"]
        kernel_size = conv_blk_opt["conv"]["out_channels"]
        leaky_relu_slope = conv_blk_opt["leaky_relu"]["slope"]
        innorm_num_features = conv_blk_opt["innorm"]["num_features"]
        num_blk = opt["num_blk"]
        self.model = []

        for n in range(num_blk):
            if n == 0:
                self.model += [
                    nn.utils.spectral_norm(nn.Conv2d(
                        in_channels[n], out_channels[n], kernel_size)),
                    nn.LeakyReLU(leaky_relu_slope)
                    ]
            else:
                self.model += [
                    nn.utils.spectral_norm(nn.Conv2d(
                        in_channels[n], out_channels[n], kernel_size)),
                    nn.InstanceNorm2d(innorm_num_features[n]),
                    nn.LeakyReLU(leaky_relu_slope)
                    ]
        self.model += [
            nn.utils.spectral_norm(nn.Conv2d(
                opt["last_layer"]["in_channels"],
                opt["last_layer"]["out_channels"]
                opt["last_layer"]["kernel_size"]            
                )),
            nn.Sigmoid()
            ]
        self.model = nn.sequential(self.model)


    def forward(self, x):
        return self.model(x)