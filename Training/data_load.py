import sys
import os
import re
import glob
from   enum import Enum
import random
import numpy as np
import torch
from   torch.utils.data import Dataset
from   skimage import io, transform
from   collections import namedtuple
# To display images:
#from   skimage.viewer import ImageViewer
#viewer = ImageViewer(image)
#viewer.show()



class Process(Enum):
    Train = 1
    Dev   = 2
    Test  = 3



SubSet = namedtuple( 'SubSet', [ 'coords_dir', 'imgs_dir', 'n_prev' ] )



class GazeDataset( Dataset ):

    def __init__( self, base_dir, transform=None, n_dev=2000, n_test=2000, target_res=(320,240) ):
        self._base_dir     = base_dir
        self._sub_sets     = []
        self._n_images     = 0
        self._sd_n_images  = []
        self._all_indexes  = []
        self._image_paths  = []
        self._n_train      = 0
        self._n_dev        = 0
        self._n_test       = 0
        self._process      = Process.Train
        self._transform    = transform

        # Target resolution
        assert isinstance( target_res, tuple )
        w_t, h_t = target_res
        a_t      = w_t*h_t

        for sd in os.listdir( self._base_dir ) :
            # Consider only sub-directories
            sd_path = os.path.join( self._base_dir, sd )
            if not os.path.isdir( sd_path ) :
                continue
            # We may not have created sub-sub-directories to store resized images
            imgs_dir = sd
            # Find the sub-sub-directory with resolutions the closest to the objective
            best = sys.maxsize
            reg  = re.compile( '\d+x\d+' )
            for ssd in os.listdir( sd_path ) :
                if not reg.fullmatch( ssd ) :
                    continue
                ssd_path = os.path.join( sd_path, ssd )
                if not os.path.isdir( ssd_path ) :
                    continue
                w, h = ssd.split( 'x' )
                diff = abs( a_t - int(w)*int(h) )
                if diff < best :
                    best      = diff
                    imgs_dir  = os.path.join( sd, ssd )

            img_glob  = os.path.join( self._base_dir, imgs_dir, '*.png' )
            img_names = glob.glob( img_glob )
            n_imgs    = len( img_names )

            # Check number of images is equal to the number of lines in coordinates file
            n_coordinates = 0
            with open( os.path.join( sd_path, 'coordinates.csv' ), 'r' ) as f:
                for line in f:
                    n_coordinates += 1
            if ( n_imgs != n_coordinates ) :
                raise ValueError( 'Error: different number of images and coordinates for directory '
                                  + imgs_path )

            # It's all good
            self._sd_n_images.append( n_imgs )
            self._sub_sets.append( SubSet( sd, imgs_dir, self._n_images ) )
            self._n_images += n_imgs

        self._all_indexes = list( range( 0, self._n_images ) )
        random.seed( 0 )
        random.shuffle( self._all_indexes )

        if ( self._n_images > ( n_dev + n_test ) ) :
            self._n_train = self._n_images - ( n_dev + n_test )
            self._n_dev   = n_dev
            self._n_test  = n_test


    def __len__( self ):
        n = 0
        if ( Process.Train == self._process ) : n = self._n_train
        if ( Process.Dev   == self._process ) : n = self._n_dev
        if ( Process.Test  == self._process ) : n = self._n_test
        #if ( Process.Train == self._process ) : n = 800
        #if ( Process.Dev   == self._process ) : n = 80
        #if ( Process.Test  == self._process ) : n = 80
        return n


    def __getitem__( self, idx ):
        # Basic validity check
        max_index = self._n_train - 1
        if ( Process.Dev  == self._process ) : max_index = self._n_dev  - 1
        if ( Process.Test == self._process ) : max_index = self._n_test - 1
        if ( idx > max_index ) :
            raise ValueError( 'Error: requested data index out of range' )

        # Adjust index
        if ( Process.Dev   == self._process ) : idx += self._n_train
        if ( Process.Test  == self._process ) : idx += self._n_train + self._n_dev
        idx = self._all_indexes[ idx ]

        # Find subset and compute local index
        n_sub_sets = len( self._sub_sets )
        s_idx      = n_sub_sets - 1

        # Linear search
        #for ii in range( 0, n_sub_sets - 1 ) :
        #    if self._sub_sets[ii+1].n_prev > idx :
        #        s_idx = ii
        #        break

        # Binary search
        fst = 0
        lst = n_sub_sets - 1
        while lst > fst+1 :
            mid = (fst+lst)//2
            if idx < self._sub_sets[mid].n_prev :
                lst = mid
            else :
                fst = mid
        if ( idx < self._sub_sets[fst+1].n_prev ) or ( 1 == n_sub_sets ):
            s_idx = fst
        else:
            s_idx = lst

        local_id = idx - self._sub_sets[s_idx].n_prev + 1 # saved data start from 1
        base_dir = self._base_dir
        subset   = self._sub_sets[s_idx]

        # Read image
        image_name = os.path.join( base_dir, subset.imgs_dir, str(local_id)+'.png' )
        image      = io.imread( image_name )

        # Read coordinates
        coords = np.matrix( [0,0], dtype=float )
        found  = False
        with open( os.path.join( base_dir, subset.coords_dir, 'coordinates.csv' ), 'r' ) as f:
            for line in f:
                values = line.split( ';' )
                if local_id == int( values[0] ):
                    coords = np.matrix( [ float( values[1] ), float( values[2] ) ], dtype=float )
                    found  = True
                    break

        if ( not found ):
            raise ValueError( 'Error: coordinates not found' )

        item = {'image': image, 'coords': coords}

        if self._transform:
            item = self._transform( item )

        return item


    def setProcess( self, process ):
        self._process = process



class ScaleImage( object ):
    """Rescale the image in a sample to a given size, and normalize it.

    Args:
        output_size ( tuple ): Desired output size.
    """
    def __init__( self, output_size ):
        assert isinstance( output_size, tuple )
        self._output_size = output_size


    def __call__( self, sample ):
        image, coords = sample['image'], sample['coords']
        h    , w      = image.shape[:2]
        new_w, new_h  = self._output_size
        new_w, new_h  = int(new_w), int(new_h)

        if  ( new_h != h ) or ( new_w != w ) :
            if 2 == image.ndim :
                res = transform.resize( image, (new_h,new_w) )
            else :
                res = np.ndarray( (new_h,new_w,image.shape[2]), image.dtype )
                for i in range( 0, image.shape[2] ) :
                    res[:,:,i] = transform.resize( image[:,:,i], (new_h,new_w) )
        else:
            res = image.astype('float64')/255 # to be consistent, as transform.resize results ∈ [0,1]

        return {'image': res, 'coords': coords}



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



class CenterPixels( object ):
   """Center the image ( channels ) values
   """

   def __call__( self, sample ):
       image, coords = sample['image'], sample['coords']

       if 2 == image.ndim :
           #normed = ( image - image.mean() ) / image.std()
           centered = image - 0.5
       else :
           for i in range( 0, image.shape[2] ):
               #normed = ( image[:,:,i] - image[:,:,i].mean() ) / image[:,:,i].std()
               centered = image[:,:,i] - 0.5

       return {'image': centered, 'coords': coords}



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
