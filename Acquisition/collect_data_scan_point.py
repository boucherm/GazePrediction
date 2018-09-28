import os
import sys
import math
import random
import os.path
import datetime
import time
from   threading import Thread, Lock
import numpy as np
import cv2
import PyQt5           as pyqt
import PyQt5.QtCore    as qtcore
import PyQt5.QtWidgets as qtwidgets
import PyQt5.QtGui     as qtgui
from   PyQt5.QtCore    import QPoint  , QRect        , QSize    , Qt     , pyqtProperty , pyqtSignal
from   PyQt5.QtWidgets import QWidget , QApplication , QLabel
from   PyQt5.QtGui     import QPixmap , QImage       , QPainter , QBrush, QColor

import faulthandler
faulthandler.enable()

# Notes:
# OpenCV:
#    Print content of cv2 module
#       print( dir( cv2 ) )
# Qt:
#    Examples: /usr/share/doc/pyqt5-examples/examples
#    Help in interpreter:
#       import PyQt5
#       import PyQt5.QtGui
#       help( PyQt5.QtGui.QImage )


class Recorder( Thread ):

    _cap           = cv2.VideoCapture
    _gray          = np.zeros(1)
    _n_u_cells     = 10;
    _n_v_cells     = 10;
    _u_cell        = 0;
    _v_cell        = -1;
    _u             = 0;
    _v             = 0;
    _data_dir_name = ""
    _data_counter  = 0
    _is_running    = False
    _exec_lock     = Lock()
    _is_paused     = True
    _is_going_down = True
    _n_shots       = 5


    def __init__( self, screen_width, screen_height, margin, widget ):
        Thread.__init__( self )
        self._cap           = cv2.VideoCapture( 0 )
        ret, frame          = self._cap.read()
        self._gray          = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
        self._data_dir_name = '../Data/' + datetime.datetime.now().strftime( "%Y_%m_%d-%H_%M_%S/" )
        self._screen_width  = screen_width;
        self._screen_height = screen_height;
        self._margin        = margin
        self._u_low         = self._margin
        self._v_low         = self._margin
        self._u_high        = self._screen_width  - 1 - self._margin
        self._v_high        = self._screen_height - 1 - self._margin
        self._widget        = widget


    def __del__( self ):
        self._cap.release()


    def run( self ):
        time.sleep( 1.0 )

        # Bootstrap display
        self.select_coordinates()
        self._widget.set_coordinates( self._u, self._v )
        self._widget.draw_tile()
        self._widget.display()

        with self._exec_lock:
            self._is_running = True
            self._is_paused  = True
            is_running       = self._is_running
            is_paused        = self._is_paused

        #print( '---> before run' )
        while is_running:
            #print( '---> running' )
            if is_paused:
                #print( '--->   paused' )
                time.sleep( 0.05 ) # XXX using a condition variable would be better
            else:
                #print( '--->   playing' )
                self._widget.draw_tile()
                self._widget.display()

                # TODO "wait" for confirmation that display was updated

                # For some reason, I need to "dequeue" the camera
                for ii in range( 0, 5 ):
                    ret, frame = self._cap.read()

                for ii in range( 0, self._n_shots ):
                    time.sleep( 0.035 )
                    ret, frame = self._cap.read()
                    self._gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
                    self.save_data()

                self._widget.clear_tile()
                is_over = self.select_coordinates()
                if is_over :
                    #print( "--->   is over " )
                    self._widget.close()
                    break

                self._widget.pause()
                self._widget.set_coordinates( self._u, self._v )
                self._widget.draw_tile()
                self._widget.display()
                self.pause()

            with self._exec_lock:
                is_running = self._is_running
                is_paused  = self._is_paused
        #print( '---> end of run' )


    def select_coordinates( self ):
        #print( "--- select_coordinates" )
        is_over  = False
        if self._is_going_down:
            if ( self._v_cell + 1 ) < self._n_v_cells :
                self._v_cell += 1
            else :
                if ( self._u_cell + 1 ) < self._n_u_cells :
                    self._u_cell += 1
                    self._is_going_down = False
                else :
                    is_over = True
        else :
            if ( self._v_cell - 1 ) >= 0 :
                self._v_cell -= 1
            else :
                if ( self._u_cell + 1 ) < self._n_u_cells :
                    self._u_cell += 1
                    self._is_going_down = True
                else :
                    is_over = True

        #print( "X: ", X )
        #print( "self._v_cell: ", self._v_cell )
        #print( "self._u_cell: ", self._u_cell )
        v_length = self._screen_height / self._n_v_cells
        u_length = self._screen_width  / self._n_u_cells
        #print( "v_length: ", v_length )
        #print( "u_length: ", u_length )
        self._v  = v_length*self._v_cell + random.randint( 0, v_length - 1 )
        self._u  = u_length*self._u_cell + random.randint( 0, u_length - 1 )
        #print( "self._u: ", self._u )
        #print( "self._v: ", self._v )

        return is_over


    def save_data( self ):
        os.makedirs( name=self._data_dir_name, mode=0o775, exist_ok=True )
        self._data_counter += 1
        with open( self._data_dir_name + '/coordinates.csv', 'a' ) as f:
            u = self._u
            v = self._v
            f.write( str( self._data_counter ) + '; ' + str( u ) + '; ' + str( v ) + '\n' )
        f.closed
        cv2.imwrite( self._data_dir_name + '/' + str( self._data_counter ) + '.png', self._gray )


    def play( self ):
        with self._exec_lock:
            self._is_paused = False


    def pause( self ):
        with self._exec_lock:
            self._is_paused = True


    def stop( self ):
        #print( '---> stopped' )
        with self._exec_lock:
            self._is_paused  = True
            self._is_running = False



# Overload the widget class to get access to keyboard's pressed keys
class Widget( QWidget, Thread ):

    _image         = QImage
    _recorder      = Recorder
    _label         = QLabel
    _tile_size     = 31;
    _screen_width  = 1920;
    _screen_height = 1080;
    _u             = 0;
    _v             = 0;
    _is_recording  = False;
    _margin        = ( _tile_size - 1 ) / 2.0
    _exec_lock     = Lock()
    _coords_lock   = Lock()


    def __init__( self ):
        super( Widget, self ).__init__()
        self._image    = QImage( self._screen_width, self._screen_height, QImage.Format_RGB32 )
        self._label    = QLabel
        self._recorder = Recorder( self._screen_width, self._screen_height, self._margin, self )
        self._recorder.start()


    def setLabel( self, label ):
        self._label = label


    def pause( self ): # TODO better name
        with self._exec_lock:
          self._is_recording = False


    def keyPressEvent( self, e ):
        #print( '___o keypress' )
        if ( Qt.Key_Escape == e.key() ) or ( Qt.Key_Q == e.key() ):
            #print( '___o escape or q' )
            self._recorder.stop()
            self._recorder.join()
            self.close()

        if Qt.Key_Space == e.key():
            #print( '___o space' )
            with self._exec_lock:
              self._is_recording = True
            self._recorder.play()


    def set_coordinates( self, u, v ):
        with self._coords_lock:
            self._u = u
            self._v = v


    def draw_tile( self ):
        with self._coords_lock:
            u = self._u
            v = self._v

        with self._exec_lock:
          recording = self._is_recording

        color = Qt.black if ( recording ) else QColor( 30, 30, 30 )

        painter = QPainter( self._image )
        painter.fillRect( QRect( u - self._margin,
                                 v - self._margin,
                                 self._tile_size,
                                 self._tile_size ),
                          QBrush( color ) )
        painter.fillRect( QRect( u - 5 , v - 5 , 10 , 10 ) , QBrush( Qt.black ) )
        painter.fillRect( QRect( u     , v     , 1  , 1 )  , QBrush( QColor(0,100,255) ) )


    def display( self ):
        self._label.setPixmap( QPixmap.fromImage( self._image ) )
        self.showFullScreen()
        self.update()


    def clear_tile( self ):
        #print( "___o clear_tile " )
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
    w.setLabel( label )
    #w.start()
    w.run()
    sys.exit( app.exec_() )
