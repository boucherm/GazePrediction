import os
import glob
import numpy as np
import torch
from   torch.utils.data import Dataset
from   skimage import io, transform
from   enum import Enum



class DataChannels(Enum):
    One = 1
    All = 2



class GazeDataset( Dataset ):

    def __init__( self, base_dir, channels, transform=None ):
        self._base_dir    = base_dir
        self._sub_dirs    = os.listdir( self._base_dir )
        self._n_images    = 0
        self._sd_n_images = []
        self._channels    = channels

        for sd in self._sub_dirs:
            images_glob     = base_dir + "/" + sd + "/*.png"
            images_names    = glob.glob( images_glob )
            self._n_images += len( images_names )
            self._sd_n_images.append( len( images_names ) )

        self.transform = transform


    def __len__( self ):
        return self._n_images if ( DataChannels.One == self._channels ) else int( self._n_images/5 )
        #return 30*16;
        #return 100;


    def __getitem__( self, idx ):
        # Note:
        #    The whole thing could be way more efficient
        #    I'm just not focusing on that at the moment
        sd_index = 0
        counter  = 0
        raw_id   = idx if ( DataChannels.One == self._channels ) else int( idx*5 )
        for n in self._sd_n_images:
            if counter + n > raw_id :
                break
            else:
                counter += n
                sd_index += 1

        local_id = raw_id - counter + 1
        sd_path  = os.path.join( self._base_dir, self._sub_dirs[sd_index] )

        image_name = os.path.join( sd_path, str(local_id) + ".png" )
        image_one  = io.imread( image_name )
        if DataChannels.One == self._channels :
            image = image_one
        else :
            # skimage.io.concatenate_images may be an alternative
            h, w  = image_one.shape[:2]
            image = np.ndarray( (h,w,5), image_one.dtype )
            image[ :, :, 0 ] = image_one
            for i in range( 1, 5 ):
                image_name       = os.path.join( sd_path, str(local_id+i) + ".png" )
                image[ :, :, i ] = io.imread( image_name )

        coords = np.matrix( [0,0], dtype=float )
        with open( os.path.join( sd_path, 'coordinates.csv' ), 'r' ) as f:
            for line in f:
                values = line.split( ';' )
                if local_id == int( values[0] ):
                    coords = np.matrix( [ float( values[1] ), float( values[2] ) ], dtype=float )
                    break

        item = {'image': image, 'coords': coords}

        if self.transform:
            item = self.transform( item )

        return item



class ScaleImage( object ):
    """Rescale the image in a sample to a given size.

    Args:
        output_size ( tuple ): Desired output size.
    """
    def __init__( self, output_size ):
        assert isinstance( output_size, tuple )
        self._output_size = output_size


    def __call__( self, sample ):
        image, coords = sample['image'], sample['coords']
        h    , w      = image.shape[:2]
        new_h, new_w  = self._output_size
        new_h, new_w  = int(new_h), int(new_w)

        if 2 == image.ndim :
            resized = transform.resize( image, (new_h,new_w) )
        else :
            resized = np.ndarray( (new_h,new_w,5), image.dtype )
            for i in range( 0, 5 ) :
                resized[:,:,i] = transform.resize( image[:,:,i], (new_h,new_w) )

        return {'image': resized, 'coords': coords}



class NormalizeCoordinates( object ):
    """Normalize the target coordinates

    Args:
        screen_size ( tuple ): screen resolution ( width x height )
    """
    def __init__( self, screen_size ):
        assert isinstance( screen_size, tuple )
        self._screen_size = screen_size
        self._half_size   = np.matrix( [ screen_size[0]/2.0, screen_size[1]/2.0 ], dtype=float )


    def __call__( self, sample ):
        image, coords = sample['image'], sample['coords']
        coords        = np.divide( coords - self._half_size, self._half_size )
        return {'image': image, 'coords': coords}



class ToTensor( object ):
    """Convert ndarrays in item to Tensors."""

    def __call__( self, item ):
        image, coords = item['image'], item['coords']

        if 2 == image.ndim :
            # Grayscale image
            image = np.expand_dims( image, 0 )
        else :
            # For rgb images swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            image = image.transpose( ( 2, 0, 1 ) )

        return { 'image': torch.from_numpy( image ), 'coords': torch.from_numpy( coords ) }
