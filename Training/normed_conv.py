import torch
import torch.nn as nn



# An "absoluted" then normalized conv from a standard conv layer, to carry coordinates transformation
class NormedConv( nn.Conv2d ):

    def __init__( self, c_in, c_out, k, s, b=False ):
        super().__init__( in_channels=c_in, out_channels=c_out, kernel_size=k, stride=s, bias=b )


    def set_weights( self, weights ) :
        for f in range( 0, weights.shape[0] ):
            weights_abs          = torch.abs( weights[f,:,:,:] )
            sum_weights_abs      = torch.sum( weights_abs )
            self.weight[f,:,:,:] = torch.div( weights_abs, sum_weights_abs )
