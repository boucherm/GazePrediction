import torch
import torch.nn as nn
import torch.nn.functional as F



# TODO:
#   . add batchnorm1D
#   . add dropout1D

class GazeNet320x240( nn.Module ):

    def __init__( self ):
        super( GazeNet320x240, self ).__init__()
        # layer1
        self.conv1  = nn.Conv2d(  1, 16, 7 )
        self.batch1 = nn.BatchNorm2d( 16 )
        self.drop1  = nn.Dropout2d( p=0.2 )
        # layer2
        self.conv2  = nn.Conv2d( 16,  16, 7 )
        self.batch2 = nn.BatchNorm2d( 16 )
        self.drop2  = nn.Dropout2d( p=0.2 )
        # layer3
        self.conv3  = nn.Conv2d(  16,  16, 7 )
        self.batch3 = nn.BatchNorm2d( 16 )
        self.drop3  = nn.Dropout2d( p=0.2 )
        # layer4
        self.conv4  = nn.Conv2d(  16,  16, 5 )
        self.batch4 = nn.BatchNorm2d( 16 )
        self.drop4  = nn.Dropout2d( p=0.2 )
        # layer5
        self.conv5  = nn.Conv2d(  16,  8, 5 )
        self.batch5 = nn.BatchNorm2d( 8 )
        self.drop5  = nn.Dropout2d( p=0.2 )
        # Fully connected
        self.fc1    = nn.Linear( 8 * 17 * 12, 816 )
        self.fc2    = nn.Linear( 816, 408 )
        self.fc3    = nn.Linear( 408, 204 )
        self.fc4    = nn.Linear( 204, 2 )

        #self.pool = nn.MaxPool2d( 2, 2 )
        #self.batch6 = nn.BatchNorm1d( 120 )
        #self.batch7 = nn.BatchNorm1d( 120 )

    def forward( self, x ):
        # 320x240
        x = F.relu( self.conv1(x) ) # 314 x 234
        x = self.batch1(x)
        x = self.drop1(x)

        x = F.relu( self.conv2(x) ) # 308 x 228
        x = self.batch2(x)
        x = self.drop2(x)
        x = F.max_pool2d( x, (2,2) )# 154 x 114

        x = F.relu( self.conv3(x) ) # 148 x 108
        x = self.batch3(x)
        x = self.drop3(x)

        x = F.relu( self.conv4(x) ) # 144 x 104
        x = self.batch4(x)
        x = self.drop4(x)
        x = F.max_pool2d( x, (2,2) )# 72 x 52

        x = F.relu( self.conv5(x) ) # 68 x 48
        x = self.batch5(x)
        x = self.drop5(x)
        x = F.max_pool2d( x, (4,4) )# 17 x 12

        x = x.view( -1, 8 * 17 * 12 )

        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        x = F.relu( self.fc3(x) )
        x = self.fc4(x)

        return x
