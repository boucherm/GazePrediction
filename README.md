This project is mainly an exercise but also an attempt at using a webcam to predict the mouse pointer position.
This is ( not ) achieved ( but not that far ) using a quite simple neural net.
As a neural net needs to be trained, training data is needed.
This project presents three python scripts: one to acquire data, one to define and train a neural net, one to test the results.


Required hardware: a webcam rigidly fixed to the screen\
Required software: python3, opencv python packages, pyqt5, scikit-image, pytorch, numpy


Configuration
=============

File: `config.txt`

Fill in your screen's width and height values.


Acquisition
===========

File: `Acquisition/collect_data_scan_point.py`

Displays a black image that covers all your screen.
A blue pixel surrounded by a grey square is displayed ( for each new position ).
While looking at the blue pixel, press `space` to acquire some images ( meanwhile the grey square is removed ).
Once a point data has been gathered a new point is selected and displayed.
Points are first sampled on a 10x10 grid, then on 40 locations along the screen border.
Once all locations have been scanned the program stops.
Press `q` or `escape` to stop early, but a complete run shouldn't be too long.

To provide some robustness to sensor noise and blinking, five images are taken for each sampled point.
As a complete acquisition run scans 10x10 + 40 positions, it produces 700 training data.
It usually takes about 2mn for me.

You should probably aim for at least 20 000 images ( therefore performing 30 acquisition runs ).
Perform different acquisition runs preferably at different: times of the day, positions in front of your screen, screen height, screen inclination, etc...
In short: try to make the training data representative of the final use cases.

Quirk: the net is prone to learn to only use the head orientation ( thus ignoring eyes ).
To counter this, you can perform some acquisition runs while:
* keeping your head still, moving only your eyes to look at the pixel/cursor
* continuously moving your head, in slow motion ( to avoid motion blur )

Adjustment to your situation:
If you have more than one webcam connected to your computer you may need to change the camera index ( default value: `0` ) in the `cv2.VideoCapture(0)` call.


Training
========

File: `Training/train.py`

The net is defined in `Training/gaze_net.py`
~~To retain enough details, images are downscaled only to a 320x160 resolution.~~
You can choose the size of the image you want to process.
You'd need to adjust the convolutions parameters if you do so.
Processed images are quite big, therefore training is quite long.
With an nvidia gtx970 card, training on about 40 000 images takes about 18h.
During training the parameters are regularly saved to disk.
Resulting files have the names: `gn_epochX_str_mseY_Z.pt`.
In these names:
* `epochX` simply is the epoch number.
* `str` is `train` or `dev` as for which dataset the loss is the new best loss
* `mseY.ZZZZ` is the mean squared error for the net on the train/dev dataset

Quirks: depending on the number of training data you have, you may need or benefit from tinkering with the `batch_size` and `exp` variables.
If these values are too big the net won't learn.
If they are too small the net will be slow to learn.

Warning: expect the architecture to change.

Notes:
* While training you will see losses for train, dev and test sets ( names from Andrew Ng ).
Dev and test sets are picked randomly ( with a constant seed ) among the data folder.
If the dataset doesn't change, from one training to another, the dev and test sets don't change either.
* The reason for the neural being quite simple ( in terms of number of parameters ) is that a big one takes lot of time ( I have a modest gpu ) and data ( acquisition takes time too ) to train.


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
If you have more than one webcam connected to your computer you may need to change the camera index ( default value: `0` ) in the `cv2.VideoCapture(0)` call.

** ! The test script hasn't been updated and therefore is broken ! **
It shouldn't be very hard to restore, though.
