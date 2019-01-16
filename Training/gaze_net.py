import math
from   enum import Enum
import numpy               as np
import torch
import torch.nn            as nn
import torch.nn.functional as F
import normed_conv         as nc
import net_utils           as nu
import io_utils            as iu



# TODO:
# . See https://jhui.github.io/2018/02/09/PyTorch-neural-networks/ about model transfer
# . Spatial transformers https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
# https://arxiv.org/abs/1506.02025


class GazeNet320x240_Strided( nn.Module ):

    def __init__( self ):
        torch.manual_seed(0)
        super( GazeNet320x240_Strided, self ).__init__()
        # layer1
        self.conv1  = nn.Conv2d(  1, 16, 7 )
        self.batch1 = nn.BatchNorm2d( 16 )
        # layer2
        self.conv2  = nn.Conv2d( 16,  16, 7, 2 )
        #self.conv2  = normed_conv( 16,  16, 7, 2 )
        self.batch2 = nn.BatchNorm2d( 16 )
        # layer3
        self.conv3  = nn.Conv2d(  16,  16, 7, 2 )
        self.batch3 = nn.BatchNorm2d( 16 )
        # layer4
        self.conv4  = nn.Conv2d(  16,  16, 5, 2 )
        self.batch4 = nn.BatchNorm2d( 16 )
        # layer5
        self.conv5  = nn.Conv2d(  16,  16, 5, 2 )
        self.batch5 = nn.BatchNorm2d( 16 )
        # Layer6
        self.fc1    = nn.Linear( 16 * 16 * 11, 1632 )
        self.batch6 = nn.BatchNorm1d( 1632 )
        # Layer7
        self.fc2    = nn.Linear( 1632, 816 )
        self.batch7 = nn.BatchNorm1d( 816 )
        # Layer8
        self.fc3    = nn.Linear( 816, 408 )
        self.batch8 = nn.BatchNorm1d( 408 )
        self.fc4    = nn.Linear( 408, 204 )
        # Layer9
        self.batch9 = nn.BatchNorm1d( 204 )
        self.fc5    = nn.Linear( 204, 102 )
        # Final layer
        self.fc6    = nn.Linear( 102, 2 )


    def forward( self, x ):
        # 320x240
        x = F.leaky_relu( self.conv1(x) ) # 314 x 234
        x = self.batch1(x)
        x = F.dropout2d( x, p=0.2 )

        x = F.leaky_relu( self.conv2(x) ) # 154 x 114
        x = self.batch2(x)
        x = F.dropout2d( x, p=0.2 )

        x = F.leaky_relu( self.conv3(x) ) # 74 x 54
        x = self.batch3(x)
        x = F.dropout2d( x, p=0.2 )

        x = F.leaky_relu( self.conv4(x) ) # 35 x 25
        x = self.batch4(x)
        x = F.dropout2d( x, p=0.2 )

        x = F.leaky_relu( self.conv5(x) ) # 16 x 11
        x = self.batch5(x)
        x = F.dropout2d( x, p=0.2 )

        x = x.view( -1, 16 * 16 * 11 )

        x = F.leaky_relu( self.fc1(x) )
        x = self.batch6(x)
        x = F.dropout( x, p=0.2 )

        x = F.leaky_relu( self.fc2(x) )
        x = self.batch7(x)
        x = F.dropout( x, p=0.2 )

        x = F.leaky_relu( self.fc3(x) )
        x = self.batch8(x)
        x = F.dropout( x, p=0.2 )

        x = F.leaky_relu( self.fc4(x) )
        x = self.batch9(x)

        x = F.leaky_relu( self.fc5(x) )
        x = self.fc6(x)

        return x



#--- Image size after a max pool:
# import math
# l=68; k=4; math.floor( ((l-(k-1)-1)/k + 1 ) )
class GazeNet320x240_Pooled( nn.Module ):

    def __init__( self ):
        super( GazeNet320x240_Pooled, self ).__init__()
        # layer1
        self.conv1  = nn.Conv2d(  1, 16, 7 )
        self.batch1 = nn.BatchNorm2d( 16 )
        # layer2
        self.conv2  = nn.Conv2d( 16,  16, 7 )
        self.batch2 = nn.BatchNorm2d( 16 )
        # layer3
        self.conv3  = nn.Conv2d(  16,  16, 7 )
        self.batch3 = nn.BatchNorm2d( 16 )
        # layer4
        self.conv4  = nn.Conv2d(  16,  16, 5 )
        self.batch4 = nn.BatchNorm2d( 16 )
        # layer5
        self.conv5  = nn.Conv2d(  16,  16, 5 )
        self.batch5 = nn.BatchNorm2d( 16 )
        # Layer6
        self.fc1    = nn.Linear( 16 * 17 * 12, 1632 )
        self.batch6 = nn.BatchNorm1d( 1632 )
        # Layer7
        self.fc2    = nn.Linear( 1632, 816 )
        self.batch7 = nn.BatchNorm1d( 816 )
        # Layer8
        self.fc3    = nn.Linear( 816, 408 )
        self.batch8 = nn.BatchNorm1d( 408 )
        self.fc4    = nn.Linear( 408, 204 )
        # Layer9
        self.batch9 = nn.BatchNorm1d( 204 )
        self.fc5    = nn.Linear( 204, 102 )
        # Final layer
        self.fc6    = nn.Linear( 102, 2 )


    def forward( self, x ):
        # 320x240
        x = F.leaky_relu( self.conv1(x) ) # 314 x 234
        x = self.batch1(x)
        x = F.dropout2d( x, p=0.2 )

        x = F.leaky_relu( self.conv2(x) ) # 308 x 228
        x = self.batch2(x)
        x = F.dropout2d( x, p=0.2 )
        x = F.max_pool2d( x, (2,2) )      # 154 x 114

        x = F.leaky_relu( self.conv3(x) ) # 148 x 108
        x = self.batch3(x)
        x = F.dropout2d( x, p=0.2 )

        x = F.leaky_relu( self.conv4(x) ) # 144 x 104
        x = self.batch4(x)
        x = F.dropout2d( x, p=0.2 )
        x = F.max_pool2d( x, (2,2) )      # 72 x 52

        x = F.leaky_relu( self.conv5(x) ) # 68 x 48
        x = self.batch5(x)
        x = F.dropout2d( x, p=0.2 )
        x = F.max_pool2d( x, (4,4) )      # 17 x 12

        x = x.view( -1, 16 * 17 * 12 )

        x = F.leaky_relu( self.fc1(x) )
        x = self.batch6(x)
        x = F.dropout( x, p=0.2 )

        x = F.leaky_relu( self.fc2(x) )
        x = self.batch7(x)
        x = F.dropout( x, p=0.2 )

        x = F.leaky_relu( self.fc3(x) )
        x = self.batch8(x)
        x = F.dropout( x, p=0.2 )

        x = F.leaky_relu( self.fc4(x) )
        x = self.batch9(x)

        x = F.leaky_relu( self.fc5(x) )
        x = self.fc6(x)

        return x



#-------------------------------------------------------------------------------
# Convolution parameters
class CP():
    def __init__( self, k, s, c ):
        self._k = k                # kernel size
        self._s = s                # stride
        self._c = c                # n channels



# Absolute or relative enum
class AOR(Enum):
    Rel = 1
    Abs = 2



# Fully connected parameters
class FCP():
    def __init__( self, aor, v=1 ):
        self._aor = aor              # absolute or relative number of neurons
        self._v   = v                # value



# Create two tensors to store (u,v) coordinates transformations
def create_u_v_tensors( N, w, h ) :
    # Images are laid as ( N x c x h x w )
    us = np.tile( 2*np.array(range(0,h))/h - 1, (w,1) ).T
    vs = np.tile( 2*np.array(range(0,w))/w - 1, (h,1) )

    # Add c and N dims
    us = np.expand_dims( np.expand_dims(us,0), 0 )
    vs = np.expand_dims( np.expand_dims(vs,0), 0 )
    Us = np.ndarray( [ N, 1, h, w ], us.dtype )
    Vs = np.ndarray( [ N, 1, h, w ], vs.dtype )
    for ii in range( 0, N ):
        Us[ ii, :, :, : ] = us
        Vs[ ii, :, :, : ] = vs
    us = torch.from_numpy( Us ).float()
    vs = torch.from_numpy( Vs ).float()
    return ( us, vs )



#TODO
#   . ResBlocks ( maybe try to learn the weights of the sum )
#     https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#   . try different initialization ( https://pytorch.org/docs/master/nn.html#torch-nn-init )
#     or directly write tensors' data ( https://pytorch.org/docs/master/tensors.html#torch.Tensor )
#     ex: conv1.weight.data.fill_(0.01)
#   . for conv layers, add more filters for each new layer
#     ( high level features come in more variety than low level ones )
#   . use adaptive maxpooling on the last conv layer ( just for fun, try to reduce to a 1x1xn tensor )
#      -> adaptive average pooling on the last conv layer and concatenate with the maxpool
#   . Add a Kullback-Leibler divergence loss to strengthen gradient ( KLDivLoss )
class GazeNetProgressive( nn.Module ):
    """ A conv layers followed by fully connected layers neural net
          . init takes a list of conv layers parameters and a list of fc layers parameters
          . fc layers are learnt progressively, i.e. calls to increase_capacity() increase the number
            of these layers used. By default, only one is used.
          . Conv layers can be frozen, so that learning new fc layers won't mess with those
          . Can transfer coordinates as transformed by conv layers to use those in the fc layers
            ( doesn't work as well as I hoped )
    """

    # The last value of fcps should be an absolute one
    def __init__( self, width_in, height_in, cps, fcps, transfer_coords=False, N=0 ):
        super().__init__()

        self._capacity = 1 # number of fc layers applied after convs
        self._p_conv   = 0.0
        self._p_fc     = 0.0
        self._n        = 0   # number of neurons after convs
        self._string   = ''
        self._convs    = nn.ModuleList()
        self._n_convs  = []
        self._batchs2d = nn.ModuleList()
        self._fcs_in   = nn.ModuleList()
        self._batchs1d = nn.ModuleList()
        self._fcs_out  = nn.ModuleList()
        self._transfer = transfer_coords
        self._N        = N

        self.reset_dropouts()

        if self._transfer :
            us, vs = create_u_v_tensors( N, width_in, height_in )
            device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
            self._us = us.to( device )
            self._vs = vs.to( device )

        # Conv layers
        w = width_in
        h = height_in
        c = 1
        for cp in cps :
            conv  = nn.Conv2d( c, cp._c, cp._k, cp._s, bias=False )
            if self._transfer :
                n_conv = nc.NormedConv( c, cp._c, cp._k, cp._s )
                n_conv.weight.data.normal_( 0.0, np.sqrt( 1/(cp._k*cp._k) ) )
                n_conv = n_conv.to( device )
                self._n_convs.append( n_conv )
            batch = nn.BatchNorm2d( cp._c )
            self._convs.append( conv )
            self._batchs2d.append( batch )
            w, h = nu.dims_after_conv2d( w, h, cp._k, cp._s )
            c    = cp._c

        iu.tee( 'len convs: ' + str( len( self._convs ) ) )

        if self._transfer:
            # 2nd: 1x1 conv layers to "merge" features and coordinates
            #self._merge_convs = nn.ModuleList()
            #for ii in range( 0, c ):
            #    merge_conv = nn.Conv2d( 3, 2, 1, 1 )
            #    merge_conv = merge_conv.to( device )
            #    self._merge_convs.append( merge_conv )
            # 3rd:
            self._merge_conv = nn.Conv2d( 3*c, 2*c, 1, 1 )

        # Fc layers
        # 1st
        #self._n = c*w*h * ( 1 if not self._transfer else 3 )
        # 2nd & 3rd
        self._n = c*w*h * ( 1 if not self._transfer else 2 )
        n       = self._n
        iu.tee( "after convs, w: " + str(w) )
        iu.tee( "after convs, h: " + str(h) )
        iu.tee( "after convs, c: " + str(c) )
        iu.tee( "after convs, n: " + str(n) )

        last = fcps[-1]
        if AOR.Abs != last._aor :
            raise ValueError( 'Error: the last layer should be absolute, but is relative.' )

        n_lst = last._v
        for ii, fcp in enumerate( fcps, 0 ) :
            # Exit layers
            linear = nn.Linear( n, n_lst )
            linear.weight.data.normal_( 0.0, np.sqrt( 1/(n) ) )
            self._fcs_out.append( linear )
            # Intermediate layers
            if len(fcps)-1 != ii :
                aor    = fcp._aor
                n_out  = max( n_lst, ( math.floor( n*fcp._v ) if AOR.Rel == aor else fcp._v ) )
                batch  = nn.BatchNorm1d( n_out )
                linear = nn.Linear( n, n_out )
                linear.weight.data.normal_( 0.0, np.sqrt( 1/(n) ) )
                self._fcs_in.append( linear )
                self._batchs1d.append( batch )
                iu.tee( 'ii: '       + str(ii)
                      + ' | n: '     + str(n)
                      + ' | n_out: ' + str(n_out)
                      + ' | n_lst: ' + str(n_lst) )
                n = n_out


    def forward( self, x ):
        for ii in range( 0, len( self._convs ) ):
            #x = F.leaky_relu( self._convs[ii](x) )
            #x = self._batchs2d[ii](x)
            #x = F.dropout2d( x, p=self._p_conv )
            x = F.selu( self._convs[ii](x) )
            x = self._batchs2d[ii](x) # seems to work better with batch norm ¯\_(ツ)_/¯
            x = F.alpha_dropout( x, p=self._p_conv )

        if self._transfer:
            with torch.no_grad():
                u = self._us
                v = self._vs
                for ii in range( 0, len( self._n_convs ) ):
                    w = self._convs[ii].weight
                    self._n_convs[ii].set_weights( w )
                    u = self._n_convs[ii](u)
                    v = self._n_convs[ii](v)
            # 1st: just concatenate
            #x = torch.cat( ( x, u, v ), 1 )
            # 2nd: attempt at using 1x1 convs on each channel of [x,u,v] images
            #n, c, h, w = x.shape
            #x_merged = torch.zeros( ( n, 2*c, h, w ),
            #                        dtype=x.dtype,
            #                        device=torch.device('cuda:0'),
            #                        requires_grad=True )
            #print( 'x_merged shape ', x_merged.shape )
            #for ii in range( 0, c ):
            #    temp = torch.cat( ( x[:,ii:ii+1,:,:], u[:,ii:ii+1,:,:], v[:,ii:ii+1,:,:] ), 1 )
            #    print( 'temp shape ', temp.shape )
            #    print( 'ii ', ii )
            #    #x_merged[ :, 2*ii:2*ii+1,   :, : ] = self._merge_convs[2*ii]( temp )
            #    #x_merged[ :, 2*ii+1:2*ii+2, :, : ] = self._merge_convs[2*ii+1]( temp )
            #    x_merged[ :, 2*ii:2*ii+2,   :, : ] = self._merge_convs[ii]( temp )
            # 3rd: concatenate and merge
            x = torch.cat( ( x, u, v ), 1 )
            x = self._merge_conv(x)

        x = x.view( -1, self._n )

        n_in = self._capacity - 1

        for ii in range( 0, n_in ):
            #x = F.leaky_relu( self._fcs_in[ii](x) )
            #if ( ii != n_in-1 ) :
                #x = self._batchs1d[ii](x)
                #x = F.dropout( x, p=self._p_fc )
            x = F.selu( self._fcs_in[ii](x) )
            if ( ii != n_in-1 ) :
                x = self._batchs1d[ii](x)
                x = F.alpha_dropout( x, p=self._p_fc )

        x = self._fcs_out[ n_in ]( x )

        return x


    def normalize( self ):
        self.normalize()


    def increase_capacity( self, n=1 ):
        self._capacity = min( ( self._capacity + n ) , self.max_capacity() )
        iu.tee( 'Number of intermediate layers: ' + str( self._capacity - 1 ) )


    def freeze_convs( self, freeze=True ):
        requires_grad = not freeze
        iu.tee( ( 'Unfreeze' if requires_grad else 'Freeze' ) + ' conv layers' )
        with torch.no_grad(): # XXX not sure if it is necessary
            for conv in self._convs :
                for param in conv.parameters():
                    param.requires_grad = requires_grad


    def capacity( self ):
        return self._capacity


    def max_capacity( self ):
        return ( len( self._fcs_in ) + 1 )


    def reset_dropouts( self ):
        self._p_conv = 0.3
        self._p_fc   = 0.1
        iu.tee( 'Reset dropouts | p_conv: ' + str(self._p_conv) + ' | p_fc: ' + str(self._p_fc) )


    def increase_dropouts( self ):
        p_c = self._p_conv
        p_f = self._p_fc

        if p_c >= 0.8 and p_f >= 0.7 :
            return False

        if ( p_c < p_f + 0.2 ) :
            self._p_conv = min( 0.1 + self._p_conv, 1.0 )
        else :
            self._p_fc   = min( 0.1 + self._p_fc, 1.0 )

        iu.tee( 'Increase dropouts | p_conv: ' + str(self._p_conv) + ' | p_fc: ' + str(self._p_fc) )

        return True
