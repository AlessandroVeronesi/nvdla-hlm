
import numpy as np
import yaml


class nvdla:
    def __init__(self, config):
        self.config  = config

        with open(config, 'r') as f:
            specs = yaml.load(f, Loader=yaml.SafeLoader)

            self.atomicc   = specs['cmac']['atomic-c']
            self.atomick   = specs['cmac']['atomic-k']
            self.batchsize = specs['cmac']['batch-size']
        
    def getHyperParams(stride=(1,1), padding=(0,0), dilation=(1,1)):
        if isinstance(padding, tuple):
            padding_h, padding_w = padding
        else:
            padding_h = padding
            padding_w = padding

        if isinstance(dilation, tuple):
            dilation_h, dilation_w = dilation
        else:
            dilation_h = dilation
            dilation_w = dilation

        if isinstance(stride, tuple):
            stride_h, stride_w = stride
        else:
            stride_h = stride
            stride_w = stride
        
        return stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w


    def outShape(self, input, filter, stride=(1,1), padding=(0,0), dilation=(1,1)):
        (B, C, H, W) = input.shape
        (K,Ck, R, S) = filter.shape

        (stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w) = self.getHyperParams(stride, padding, dilation)

        S_ = (S -1)*dilation_w +1
        R_ = (R -1)*dilation_h +1

        W_ = (2*padding_w +W -S_)//stride_w +1
        H_ = (2*padding_h +H -R_)//stride_h +1

        return (B, K, H_, W_)

    def convolve(self, Fmap, Kmap, Bias, stride=(1,1), padding=(0,0), dilation=(1,1)):

        (B, C, H, W) = Fmap.shape
        (K,C_, R, S) = Kmap.shape

        (stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w) = self.getHyperParams(stride, padding, dilation)

        # Checks
        if (C != C_):
            raise ValueError(f'Mismatching channel sizes ({C} / {C_})')

        if (K > self.atomick):
            raise ValueError(f'K exceeding hardware resources ({K} / {self.atomick})')

        if (B > self.batchsize):
            raise ValueError(f'B exceeding hardware resources ({B} / {self.batchsize})')

        if ((dilation_h > 1) or (dilation_w > 1)):
            raise ValueError(f'Dilation > 1 not supported {dilation}')

        # Psums
        psums = np.zeros(self.outShape(Fmap, Kmap, stride, padding, dilation))
        _, _, H_, W_ = psums.shape

        # Channel OP
        for c in range(0, C, self.atomicc):
            Cend = min(C, c + self.atomicc)

            # Block OP
            for r in range(R):
                for s in range(S):

                    # Stripe OP
                    for h in range(0, H, stride_h):
                        for w in range(0, W, stride_w):

                            # Atomic OP
                            for b in range(B):

                                # Vec Loading
                                datV = Fmap[b, c:Cend, h, w]
                                wtV  = Kmap[:, c:Cend, r, s]

                                # Output Coordinates
                                oh   = h*stride_h - padding_h +r
                                ow   = w*stride_w - padding_w +s

                                psums[b, :, oh, ow] += np.matmul(datV[:,np.newaxis], wtV).flatten()
        
        # Bias Addition
        psums += Bias[None, :, None, None]

        # Exits
        return psums