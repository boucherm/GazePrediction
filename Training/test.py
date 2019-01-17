import sys
import math
import random
import os.path
import datetime
import time
import configparser
from   threading import Thread
import numpy as np
import cv2
import PyQt5           as pyqt
import PyQt5.QtCore    as qtcore
import PyQt5.QtWidgets as qtwidgets
import PyQt5.QtGui     as qtgui
from   PyQt5.QtCore    import QPoint  , QRect        , QSize    , Qt     , pyqtProperty , pyqtSignal
from   PyQt5.QtWidgets import QWidget , QApplication , QLabel   , QFileDialog
from   PyQt5.QtGui     import QPixmap , QImage       , QPainter , QBrush
import torch
import torch.nn as nn
from   skimage import io, transform



class Pointer():

    def __init__( self, screen_w, screen_h, img_w, img_h, net_path ):
        self._gray         = np.zeros(1)
        self._u            = 0;
        self._v            = 0;
        self._data_counter = 0
        self._run          = False
        self._paused       = True
        self._screen_w     = screen_w
        self._screen_h     = screen_h
        self._img_w        = img_w
        self._img_h        = img_h
        self._cap          = cv2.VideoCapture( 0 )
        print( 'Loading net..' )
        self._net          = torch.load( net_path )
        print( '...done' )
        self._dev          = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
        print( 'Moving net to GPU..' )
        self._net.to( self._dev )
        print( '...done' )
        Thread.__init__( self )


    def __del__( self ):
        self._cap.release()


    def run( self ):
        # Dequeue the camera
        for ii in range( 0, 5 ):
            _, frame = self._cap.read()
        # Acquire image
        _, frame = self._cap.read()
        gray     = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
        img_h, img_w = gray.shape
        # Resize and normalize
        if  ( img_w != self._img_w ) or ( img_h != self._img_h ) :
            image = transform.resize( gray, ( self._img_w, self._img_h ) )
            image = image - 0.5
        else :
            image = gray.astype(float)/255 - 0.5
        # Add batch and channel dimensions
        image = np.expand_dims( image, 0 )
        image = np.expand_dims( image, 0 )
        # Pass through the net
        with torch.no_grad():
            self._net.eval()
            image  = torch.from_numpy( image )
            image  = image.type( torch.FloatTensor )
            image  = image.to( self._dev )
            output = self._net( image )
            output = output.cpu().numpy()
        # Convert to screen coordinates
        half_width  = self._screen_w/2
        half_height = self._screen_h/2
        x = min( max( output[0][0], -1 ), 1 )
        y = min( max( output[0][1], -1 ), 1 )
        u = int( half_width*x  + half_width )
        v = int( half_height*y + half_height )

        return ( u, v )



class Widget( QWidget ):

    def __init__( self, screen_w, screen_h, img_w, img_h ):
        super( Widget, self ).__init__()
        self._image     = None
        self._pointer   = None
        self._label     = None
        self._tile_size = 101
        self._u         = 0
        self._v         = 0
        self._margin    = ( self._tile_size - 1 ) / 2.0
        self._screen_w  = screen_w
        self._screen_h  = screen_h
        self._image     = QImage( self._screen_w, self._screen_h, QImage.Format_RGB32 )
        net_path, _     = QFileDialog.getOpenFileName( self, 'Select net', '', 'Pytorch file (*.pt)' )
        print( 'net path: ', net_path )
        self._pointer   = Pointer( screen_w, screen_h, img_w, img_h, net_path )
        self._label     = QLabel


    def set_label( self, label ):
        self._label = label


    def start( self ):
        self.set_coordinates( -1, -1 )
        self.draw_tile()
        self.display()


    def stop( self ):
        self.clear_tile()


    def keyPressEvent( self, e ):
        if ( Qt.Key_Escape == e.key() ) or ( Qt.Key_Q == e.key() ):
            self.close()

        if Qt.Key_Space == e.key():
            self.clear_tile()
            coords = self._pointer.run()
            self.set_coordinates( coords[0], coords[1] )
            self.draw_tile()
            self.display()


    def set_coordinates( self, u, v ):
        self._u = u
        self._v = v


    def draw_tile( self ):
        u = self._u
        v = self._v

        if u < 0 or u > self._screen_w or v < 0 or v > self._screen_h :
            return

        color   = Qt.green
        painter = QPainter( self._image )
        painter.fillRect( QRect( u - self._margin,
                                 v - self._margin,
                                 self._tile_size,
                                 self._tile_size ),
                          QBrush( color ) )
        painter.fillRect( QRect( u - 5,
                                 v - 5,
                                 10,
                                 10 ),
                          QBrush( Qt.black ) )


    def display( self ):
        self._label.setPixmap( QPixmap.fromImage( self._image ) )
        self.showFullScreen()
        self.update()


    def clear_tile( self ):
        painter = QPainter( self._image )
        painter.fillRect( QRect( self._u - self._margin,
                                 self._v - self._margin,
                                 self._tile_size,
                                 self._tile_size ),
                          QBrush( Qt.black ) )



if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read( '../config.txt' )
    screen_w = int( config['SCREEN']['width'] )
    screen_h = int( config['SCREEN']['height'] )
    img_w    = int( config['IMAGE']['width'] )
    img_h    = int( config['IMAGE']['height'] )

    app   = QApplication( sys.argv )
    w     = Widget( screen_w, screen_h, img_w, img_h )
    label = QLabel( w )
    image = QImage( screen_w, screen_h, QImage.Format_RGB32 )
    label.setPixmap( QPixmap.fromImage( image ) )
    w.set_label( label )
    w.start()
    sys.exit( app.exec_() )
