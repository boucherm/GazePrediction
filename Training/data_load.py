import os
import glob
import numpy as np
import torch
from   torch.utils.data import Dataset
from   skimage import io, transform



class GazeDataset( Dataset ):

    def __init__( self, base_dir, transform=None ):
        self._base_dir    = base_dir
        self._sub_dirs    = os.listdir( self._base_dir )
        self._n_images    = 0
        self._sd_n_images = []

        for sd in self._sub_dirs:
            images_glob     = base_dir + "/" + sd + "/*.png"
            images_names    = glob.glob( images_glob )
            self._n_images += len( images_names )
            self._sd_n_images.append( len( images_names ) )

        self.transform = transform


    def __len__( self ):
        #return self._n_images
        return 30*16;


    def __getitem__( self, idx ):
      # Note:
      #    The whole thing could be way more efficient
      #    I'm just not focusing on that at the moment
      sd_index = 0
      counter  = 0
      for n in self._sd_n_images:
        if counter + n > idx :
          break
        else:
          counter += n
          sd_index += 1

      local_idx  = idx - counter + 1
      local_path = os.path.join( self._base_dir, self._sub_dirs[sd_index] )
      image_name = os.path.join( local_path, str(local_idx) + ".png" )
      image      = io.imread( image_name )

      coords = np.matrix( [ 0, 0 ], dtype=float )
      with open( os.path.join( local_path, 'coordinates.csv' ), 'r' ) as f:
        for line in f:
          values = line.split( ';' )
          if local_idx == int( values[0] ):
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
        new_h, new_w  = int( new_h ), int( new_w )

        resized = transform.resize( image, ( new_h, new_w ) )

        return {'image': resized, 'coords': coords}



class NormalizeCoordinates( object ):
    """Normalize the target coordinates

    Args:
        screen_size ( tuple ): screen resolution ( width x height )
    """
    def __init__( self, screen_size ):
        assert isinstance( screen_size, tuple )
        self._screen_size = screen_size
        self._half_size   = np.matrix( [ screen_size[0] / 2.0, screen_size[1] / 2.0 ], dtype=float )


    def __call__( self, sample ):
        image, coords = sample['image'], sample['coords']
        coords        = np.divide( coords - self._half_size, self._half_size )
        return {'image': image, 'coords': coords}



class ToTensor( object ):
    """Convert ndarrays in item to Tensors."""

    def __call__( self, item ):
        image, coords = item['image'], item['coords']

        # For rgb images swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose( ( 2, 0, 1 ) ) #
        # For grayscale images
        image = np.expand_dims( image, 0 )

        return { 'image': torch.from_numpy( image ), 'coords': torch.from_numpy( coords ) }
