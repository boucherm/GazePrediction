import sys
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



# Update the learning rate of an optimizer
def update_lr( optimizer, lr ) :
    #print( optimizer.state_dict()['param_groups'] )
    #print( optimizer.state_dict()['param_groups'][0]['lr'] )
    optimizer_state =  optimizer.state_dict()
    optimizer_state['param_groups'][0]['lr'] = lr
    optimizer.load_state_dict( optimizer_state )
    #optimizer.state_dict()['param_groups'][0]['lr'] = 2
    #print( optimizer.state_dict()['param_groups'][0]['lr'] )
    #print( optimizer.state_dict()['param_groups'] )


print( "Preparing dataset" )
data_channels = dl.DataChannels.One
#data_channels = dl.DataChannels.All
gaze_dataset = dl.GazeDataset( "../Data",
                               data_channels,
                               transform=transforms.Compose( [
                                   dl.ScaleImage( (320,240) ),
                                   dl.NormalizeCoordinates( (1920,1080) ),
                                   dl.ToTensor() ] ) )
print( "done" )

print( "Preparing dataloader" )
batch_size  = 4
#batch_size  = 1
data_loader = DataLoader( gaze_dataset, batch_size=batch_size, shuffle=True, num_workers=2 )
print( "done" )

print( "Loading net" )
net = gn.GazeNet320x240() if ( dl.DataChannels.One == data_channels ) else gn.GazeNet320x240x5()
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
init_exp             = -10 # TODO: a first pass to search for a good value of init_exp
exp                  = init_exp
optimizer            = optim.Adam( net.parameters(), lr = pow(2.0,exp) )
start_time           = time.time()
losses               = []
best_loss            = sys.float_info.max
last_save_epoch      = 0
last_lr_change_epoch = 0
epoch                = 0
n_cycles             = 3
cycle                = 1
epoch_new_cycle      = 0
# TODO add a counter and a threshold to increase the learning rate ?
while True:
    # TODO Alternative ( fastai )
    #    . multiple cycles, with a cycle corresponding to 1, k, k^2, ... epoch(s)
    #    . Decrease lr at each mini-batch, following a cosine decrease along the cycle )
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
    print( '-------------------' )
    print( '[%d] | loss: %.6f | lr: 2^%d | cycle: %d' % ( epoch, loss , exp, cycle ) )

    stop = False
    save = False

    # Save is best loss so far
    if ( loss < best_loss ):
        save      = True
        best_loss = loss

    # Save if it hasn't been in a while
    #if ( epoch >= last_save_epoch + 5 ) :
        #save = True

    # Check if lr reduction is needed, perform it if so
    e = epoch - 1
    if ( ( epoch - epoch_new_cycle ) >= 2 ) :
        delta_lr_change = ( epoch - last_lr_change_epoch )
        do_update_lr = ( ( losses[e] > 0.99*losses[e-1] ) )
                      #or ( ( delta_lr_change > 2 ) and ( losses[e] > 0.97*losses[e-2] ) ) )
        if do_update_lr :
            #print( losses[ e-2 ] )
            #print( losses[ e-1 ] )
            #print( losses[e] )
            #print( losses[e] > 0.99*losses[e-1] )
            #print( ( delta_lr_change > 2 ) and ( losses[e] > 0.97*losses[e-2] ) )
            exp -= 1 if ( delta_lr_change > 2 ) else 2
            last_lr_change_epoch = epoch
            update_lr( optimizer, lr=pow(2.0,exp) )
            print( 'learning rate => 2^', exp )

    # Check if end of a cycle is reached, stop if all cycles performed, restart otherwise
    if ( exp <= -30 ) or ( loss < 0.0001 ):
        if cycle == n_cycles :
            stop = True
            save = True
        else :
            cycle          += 1
            epoch_new_cycle = 0
            exp             = init_exp
            update_lr( optimizer, lr=pow(2.0,exp) )
            print( 'learning rate => 2^', exp )

    # Read file to check if stop has been requested
    with open( 'continue.txt', 'r' ) as f :
        # XXX ^ Not robust to simultaneous read/write, those should be exceptional though
        l    = f.readline();
        stop, save = ( True, True ) if ( "1" != l ) else ( stop, save )
    f.close()

    # Save parameters and neural net file
    if save:
        # TODO
        #   . https://pytorch.org/docs/stable/notes/serialization.html
        #   . save optimizer state ?
        #     -> https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3
        if 0 == last_save_epoch :
            os.makedirs( name=results_dir_name, mode=0o775, exist_ok=True )
            shutil.copy( "gaze_net.py", results_dir_name );
        last_save_epoch = epoch
        torch.save( net, results_dir_name + 'gn_320x240_mse'+str(loss)+'_epoch'+str(epoch)+'.pt' )
        print( 'save net' )

    # Stop
    if stop:
        break

elapsed_time = time.time() - start_time
print( 'Finished training in:', elapsed_time )
