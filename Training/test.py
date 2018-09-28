import sys
import math
import random
import os.path
import datetime
import time
from   threading import Thread
import numpy as np
import cv2
import PyQt5           as pyqt
import PyQt5.QtCore    as qtcore
import PyQt5.QtWidgets as qtwidgets
import PyQt5.QtGui     as qtgui
from   PyQt5.QtCore    import QPoint  , QRect        , QSize    , Qt     , pyqtProperty , pyqtSignal
from   PyQt5.QtWidgets import QWidget , QApplication , QLabel
from   PyQt5.QtGui     import QPixmap , QImage       , QPainter , QBrush
import torch
import torch.nn as nn
from   skimage import io, transform



class Pointer():

    _cap          = cv2.VideoCapture
    _gray         = np.zeros(1)
    _u            = 0;
    _v            = 0;
    _data_counter = 0
    _run          = False
    _paused       = True
    _width        = 1920
    _height       = 1080


    def __init__( self ):
        Thread.__init__( self )
        self._cap  = cv2.VideoCapture( 0 )
        self._net  = torch.load( 'gn.pt' )
        self._dev  = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
        self._net.to( self._dev )


    def __del__( self ):
        self._cap.release()


    def run( self ):
        for ii in range( 0, 5 ):
            ret, frame = self._cap.read()
        ret, frame = self._cap.read()
        gray       = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
        image      = transform.resize( gray, ( 320, 240 ) )
        image  = image.astype( float )
        image  = np.expand_dims( image, 0 )
        image  = np.expand_dims( image, 0 )
        with torch.no_grad():
            self._net.eval()
            image  = torch.from_numpy( image )
            image  = image.type( torch.FloatTensor )
            image  = image.to( self._dev )
            output = self._net( image )
            output = output.cpu().numpy()

        half_width  = self._width/2
        half_height = self._height/2
        x = min( max( output[0][0], -1 ), 1 )
        y = min( max( output[0][1], -1 ), 1 )
        u = int( half_width*x  + half_width )
        v = int( half_height*y + half_height )

        return ( u, v )



class Widget( QWidget ):

    _image         = QImage
    _pointer       = Pointer
    _label         = QLabel
    _tile_size     = 101;
    _screen_width  = 1920;
    _screen_height = 1080;
    _u             = 0;
    _v             = 0;
    _margin        = ( _tile_size - 1 ) / 2.0
    _u_low         = _margin
    _u_high        = _screen_width  - 1 - _margin
    _v_low         = _margin
    _v_high        = _screen_height - 1 - _margin


    def __init__( self ):
        super( Widget, self ).__init__()
        self._image   = QImage( self._screen_width, self._screen_height, QImage.Format_RGB32 )
        self._pointer = Pointer()
        self._label   = QLabel


    #def __del__( self ):


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

        if u < 0 or u > self._screen_width or v < 0 or v > self._screen_width :
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
    app   = QApplication( sys.argv )
    w     = Widget()
    label = QLabel( w )
    image = QImage( 1920, 1080, QImage.Format_RGB32 )
    label.setPixmap( QPixmap.fromImage( image ) )
    w.set_label( label )
    w.start()
    sys.exit( app.exec_() )
