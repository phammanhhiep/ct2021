from torch import nn


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.num_ds = opt["multi_scale_d"]["num_ds"]

        self.ds = []
        for n in range(self.num_ds):
            self.ds.append(PatchGANDiscriminator(
                opt["multi_scale_d"]["patch_gan"]))
            self._update_opt(n)


    def forward(self, x):
        y = []
        for n in range(self.num_ds):
            y.append(self.ds[n](x))
            self._downsample(x, n)
        return y


    def _downsample(self, x, d_index):
        """Downsample the input x, with parameters defined in self.opt. The 
        options are determined by the d_index. 
        
        Args:
            x (TYPE): Description
            d_index (TYPE): Description
        """
        

    def _update_opt(self, d_index):
        """Update the options corresponding to the d_index
        
        Args:
            d_index (TYPE): index of the target discriminator
        """


class PatchGANDiscriminator(nn.Module):

    """Implementation of the Patch-GAN described in "[2017] Image-to-image 
    translation with conditional adversarial networks"
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = 
        self.model = []
        num_blk = opt["num_blk"]

        for n in range(num_blk):
            if n == 0:
                self.model += [
                    nn.Conv2d(opt["conv"]["in_channels"][n],
                        opt["conv"]["out_channels"][n]
                        opt["conv"]["kernel_size"][n]
                        ),
                    nn.LeakyReLU()
                    ]
            else:
                self.model += [
                    nn.Conv2d(opt["conv"]["in_channels"][n],
                        opt["conv"]["out_channels"][n]
                        opt["conv"]["kernel_size"][n]),
                    nn.InstanceNorm2d(
                        opt["innorm"]["num_features"][n]),
                    nn.LeakyReLU()
                    ]
        self.model += nn.Conv2d(
            opt["last_layer"]["in_channels"],
            opt["last_layer"]["out_channels"][n]
            opt["last_layer"]["kernel_size"][n]            
            )
        self.model = nn.sequential(self.model)


    def forward(self, x):
        h = x
        for subnet in self.model:
            h = subnet(x)
            x = h
        return h   