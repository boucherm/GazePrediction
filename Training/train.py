import os
import time
import shutil
import datetime
# Ignore warnings
import warnings
warnings.filterwarnings( "ignore" )
import torch
from   torch.utils.data import DataLoader
from   torchvision import transforms#, utils
import torch.nn as nn
import torch.optim as optim
import data_load as dl
import gaze_net  as gn


print( "Preparing dataset" )
gaze_dataset = dl.GazeDataset( "../Data",
                               transform=transforms.Compose( [
                               dl.ScaleImage( (320,240) ),
                               dl.NormalizeCoordinates( (1920,1080) ),
                               dl.ToTensor()#,
                            ] ) )
print( "done" )

print( "Preparing dataloader" )
batch_size  = 4
#batch_size  = 1
data_loader = DataLoader( gaze_dataset, batch_size=batch_size, shuffle=True, num_workers=2 )
print( "done" )

print( "Loading net" )
net    = gn.GazeNet320x240()
#net    = torch.load( 'gn.pt' )
device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
net.to( device )
print( "done" )
print( device )

results_dir_name = 'Results/' + datetime.datetime.now().strftime( "%Y_%m_%d-%H_%M_%S/" )
with open( 'continue.txt', 'w' ) as f :
    f.write( "1" )
f.close()

print( 'begin training' )
criterion            = nn.MSELoss()
exp                  = -10
optimizer            = optim.Adam( net.parameters(), lr = pow(2.0,exp) )
start_time           = time.time()
losses               = []
last_save_loss       = 0
last_save_epoch      = 0
last_lr_change_epoch = 0
epoch                = 0
while True:
    print( '-----------------------------------------------------' )
    epoch += 1
    loss   = 0.0
    for i, data in enumerate( data_loader, 0 ):
        images = data[ "image" ].float()
        coords = torch.squeeze( data[ "coords" ].float(), dim=1 )
        images, coords = images.to( device ), coords.to( device )

        # zero the parameter's gradients
        optimizer.zero_grad()

        # forward, backward, update
        outputs = net( images )
        if ( i*batch_size < 4 ) or ( 0 == i*batch_size % 10000 ) :
           print( '--- ', i*batch_size, '\n');
           print( 'target:\n', coords );
           print( 'output:\n', outputs );
        batch_loss = criterion( outputs, coords )
        batch_loss.backward()
        optimizer.step()

        # print statistics
        loss = ( i*loss + batch_loss.item() ) / ( i + 1 )

    losses.append( loss )
    print( '[%d] loss: %.6f' % ( epoch, loss ), 'learning rate: 2^', exp )

    stop = False
    save = False

    if ( 1 == epoch ) or ( loss < last_save_loss / 2.0 ) or ( epoch >= last_save_epoch + 5 ):
        save = True

    e = epoch - 1
    if ( epoch >= 3 ) and ( losses[e] > 0.99*losses[e-1] ) and ( losses[e] > 0.97*losses[e-2] ) :
        exp -= 1 if ( epoch > last_lr_change_epoch + 2 ) else 2
        last_lr_change_epoch = epoch
        optimizer = optim.Adam( net.parameters(), lr=pow(2.0,exp) ) # XXX Sadly we loose momentum :(
        print( 'learning rate => 2^', exp )

    if ( exp <= -30 ) or ( loss < 0.0001 ):
        stop = True
        save = True

    with open( 'continue.txt', 'r' ) as f :
    # XXX ^ Not robust to simultaneous read/write, those shoud be exceptional though
        l    = f.readline();
        stop = True if ( "1" != l ) else stop
        save = True
    f.close()

    if save:
        # TODO
        #   . https://pytorch.org/docs/stable/notes/serialization.html
        #   . save optimizer state ?
        #     -> https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3
        if 0 == last_save_epoch :
            os.makedirs( name=results_dir_name, mode=0o775, exist_ok=True )
            shutil.copy( "gaze_net.py", results_dir_name );
        last_save_loss  = loss
        last_save_epoch = epoch
        torch.save( net, results_dir_name + 'gn_320x240_mse'+str(loss)+'_epoch'+str(epoch)+'.pt' )
        print( 'save net' )

    if stop:
        break

elapsed_time = time.time() - start_time
print( 'Finished training in:', elapsed_time )
