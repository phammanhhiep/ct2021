from torch import nn


class MultiScaleDiscriminator(nn.Module):

    """The implementation of a multi-scale discriminator described in 
    "[2018] High-Resolution Image Synthesis and Semantic Manipulation with 
    Conditional GANs (Wang et al) (NVIDIA)"
    """

    def __init__(self, opt):
        super().__init__()
        self.num_ds = opt["num_ds"]
        self.scale = opt["downsample_scale_factor"]
        self.model = []

        for n in range(self.num_ds):
            self.model.append(
                PatchGANDiscriminator(opt["PatchGANDiscriminator"]))


    def forward(self, x, x_hat):
        """
        Args:
            x (TYPE): a batch of real images 
            x_hat (TYPE): a batch of synthesized images
        
        Returns:
            TYPE: Description
        """
        pred = []
        data = [x, x_hat]

        for d in data:
            pred_i = []    
            pred_i.append(self.model[0](d))

            for n in range(1, self.num_ds):
                d = self.downsample(d)
                pred_i.append(self.model[n](d))
            pred.append(pred_i)
        return pred


    #TODO: consider argument of nn.functional.interpolate
    def downsample(self, x):
        return nn.functional.interpolate(x, 
            scale_factor=(1, 1, self.scale, self.scale))       


class PatchGANDiscriminator(nn.Module):

    """Implementation of the Patch-GAN described in "[2017] Image-to-image 
    translation with conditional adversarial networks". The output for a single
    image is not a single scalar in [0, 1], but a tensor of size (1, H, W) with 
    H*W is the number of patches of the image, whose elements in [0, 1].
    """

    #TODO: verify if spectral norm is used at the same time with instance norm, or instance norm is replaced by spectral norm
    def __init__(self, opt):
        super().__init__()
        num_conv_blks = opt["num_conv_blks"]
        in_channels = opt["in_channel"]
        out_channels_list = opt["conv"]["out_channel_list"]
        conv_kernel_size = opt["conv"]["kernel_size"]
        conv_stride = opt["conv"]["stride"]
        conv_padding = opt["conv"]["padding"]
        LeakyReLU_slope = opt["LeakyReLU_slope"]
        self.model = []

        for n in range(num_conv_blks):
            out_channels = out_channels_list[n]
            if n == 0:
                self.model = [
                    nn.utils.spectral_norm(nn.Conv2d(
                        in_channels, out_channels, conv_kernel_size, 
                        conv_stride, conv_padding)),
                    nn.LeakyReLU(LeakyReLU_slope)
                    ]
            else:
                self.model += [
                    nn.utils.spectral_norm(nn.Conv2d(
                        in_channels, out_channels, kernel_size)),
                    nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(LeakyReLU_slope)
                    ]
            in_channels = out_channels

        self.model += [
            nn.utils.spectral_norm(nn.Conv2d(
                out_channels, 1, conv_kernel_size, conv_stride, conv_padding)),
            nn.Sigmoid()
            ]
        self.model = nn.sequential(self.model)


    def forward(self, x):
        """Summary
        
        Args:
            x (TYPE): batch of input of size (B, C, H, W)
        
        Returns:
            TYPE: size (B, 1, H_p, W_p) with H_p * W_p is the number of patches 
        """
        return self.model(x)