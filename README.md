This project is both an exercise and an attempt at using a webcam to predict the mouse pointer position.
This is ( not ) achieved ( but not that far ) using a ( rather simple ) neural net.
As a neural net needs to be trained, training data is needed.
This project presents three python scripts: one to acquire data, one to define and train a neural net, one to test the results.

Required hardware: a webcam rigidly fixed to the screen\
Required software: python3, opencv python packages, pyqt5, scikit-image, pytorch, numpy



Acquisition
===========

File: `Acquisition/collect_data_scan_point.py`

Displays a black image that covers all your screen.
A blue pixel surrounded by a grey square is displayed ( for each new position ).
While looking at the blue pixel, press `space` to acquire some images ( meanwhile the grey square is removed ).
Once a point data has been gathered a new point is selected and displayed.
Points are sampled on a 10x10 grid.
Once the 10x10 grid has been scanned the program stops.
Press `q` or `escape` to stop early, but a complete run shouldn't be too long.

To provide some robustness to sensor noise and blinking, five images are taken for each sampled point.
As a complete acquisition run scans 10x10 positions, it produces 500 training data.
It usually takes between 1.5mn and 2mn for me.

You should probably aim for at least 20 000 images ( therefore performing 40 acquisition runs ).
Perform different acquisition runs preferably at different times of the day, at different positions in front of your screen, etc...
In short: try to make the training data representative of the final use cases.

Quirk: the net is prone to learn to only use the head orientation ( thus ignoring eyes ).
To counter this, it is effective to perform some acquisition runs where the head stays still and only the eyes move to look at the pixel/cursor.

Adjustment to your situation:
The screen size is hardcoded as 1920x1080.
Therefore, if your screen has a different size you should look for the `1920` and `1080` values and replace with your screen resolution values.
If you have more than one webcam connected to your computer you may need to change the camera index ( default value: `0` ) in the `cv2.VideoCapture(0)` call.


Training
========

File: `Training/train.py`

The net is defined in `Training/gaze_net.py`
To retain enough details, images are downscaled only to a 320x160 resolution.
This is quite big, therefore training is quite long.
With an nvidia gtx970 card, training on about 40 000 images takes about 10h.
During training the parameters are regularly saved to disk.
Resulting files have the names: `gn_320x240_mse0.XXXX_epochY.pt`.
In these names `mse0.XXXX` is the mean squared error for the net on the training data, and `epochY` simply is the epoch number.

Quirks: depending on the number of training data you have, you may need or benefit from tinkering with the `batch_size` and `exp` variables.
If these values are too big the net won't learn.
If they are too small the net will be slow to learn.

Adjustment to your situation:
As for the acquisition script the screen resolution is hardcoded.
Look for the `1920` and `1080` values and replace accordingly.

Warning: expect the architecture to change.
Therefore, would you be satisfied with a training results, you should save the net architecture file.


Testing
=======

File: `Training/test.py`

To access the net architecture file the test script is in the same folder as the training one.
It reads a "gn.pt" file.
Therefore, you need to symlink ( or copy and rename ) the result file you want to test as `gn.pt`.

The test script is very similar to the acquisition script.
It displays a black screen.
While looking a the screen, press `space` to capture a webcam image, and make the net predict where you were looking at.
A green square will be drawed around this predicted position.
Press `q` or `escape` to quit.

Warning: it takes quite a long time to start.

Adjustment to your situation:
As for the others scripts the screen resolution is hardcoded.
Look for the `1920` and `1080` values and replace accordingly.
