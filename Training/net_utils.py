import math



#--- Image size after conv
def dims_after_conv2d( width_in, height_in, kernel_size, stride=1 ):
    def func( l, k, s ):
        return math.floor( ( l - k )/s + 1 )
    height_out = func( height_in, kernel_size, stride )
    width_out  = func( width_in,  kernel_size, stride )

    return width_out, height_out
