import sys
import os
import time
import shutil
import datetime
import configparser
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


def tee( string, filename ) :
    print( string )
    with open( filename, 'a' ) as f:
        f.write( string )
        f.write( '\n' )



# Hyper-parameters
init_exp   = -10 # TODO: a first pass to search for a good value of init_exp
n_cycles   = 3
batch_size = 32
#batch_size = 16
#batch_size = 1

config = configparser.ConfigParser()
config.read( '../config.txt' )
screen_width  = int( config['SCREEN']['width'] )
screen_height = int( config['SCREEN']['height'] )

print( "Preparing dataset" )
data_channels = dl.DataChannels.One
#data_channels = dl.DataChannels.All
gaze_dataset = dl.GazeDataset( "../Data",
                               data_channels,
                               transform=transforms.Compose( [
                                   dl.ScaleImage( (320,240) ),
                                   dl.NormalizeCoordinates( (screen_width,screen_height) ),
                                   dl.ToTensor() ] ) )
print( "done" )

print( "Preparing dataloader" )
data_loader = DataLoader( gaze_dataset, batch_size=batch_size, shuffle=True, num_workers=2 )
print( "done" )

print( "Loading net" )
net = gn.GazeNet320x240() if ( dl.DataChannels.One == data_channels ) else gn.GazeNet320x240x5()
#net    = torch.load( 'gn_old.pt' )
device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
net.to( device )
print( "done" )
print( device )

results_dir_name = 'Results/' + datetime.datetime.now().strftime( "%Y_%m_%d-%H_%M_%S/" )
os.makedirs( name=results_dir_name, mode=0o775, exist_ok=True )
shutil.copy( "gaze_net.py", results_dir_name );
logfilename = results_dir_name + '/log.txt'
with open( 'continue.txt', 'w' ) as f :
    f.write( "1" )
f.close()

print( 'begin training' )
criterion            = nn.MSELoss()
exp                  = init_exp
optimizer            = optim.Adam( net.parameters(), lr = pow(2,exp) )
# Attempt at searching a value for initial learning rate ( exp )
# Didn't give interesting results, maybe just needs to be reworked
# ( ex: making stats on a whole epoch / on more than a single batch )
#torch.save( net, 'gn_temp.pt' )
#init_losses = []
#init_exps   = [ exp-1 ]
#for i, data in enumerate( data_loader, 0 ):
#    optimizer.zero_grad()
#    images         = data[ "image" ].float()
#    coords         = torch.squeeze( data[ "coords" ].float(), dim = 1 )
#    images, coords = images.to( device ), coords.to( device )
#    outputs        = net( images )
#    batch_loss     = criterion( outputs, coords )
#    print( str(batch_loss.item()) + " _ " + str(exp) )
#    batch_loss.backward()
#    optimizer.step()
#    init_losses.append( batch_loss.item() )
#    exp += 1
#    if exp >= 0 :
#        break
#    update_lr( optimizer, lr=pow(2.0,exp) )
#    init_exps.append( exp )
#print( len(init_losses) )
#print( len(init_exps) )
#init_exp = 31;
#best     = init_losses[1] - init_losses[0]
#for i in range( 1, len( init_losses ) ):
#    decr = init_losses[i] - init_losses[i-1]
#    print( str(i) + ": " + str(init_exps[i]) + " -> " + str(init_losses[i]) + " | " + str(decr) )
#    if decr < best :
#        init_exp = init_exps[ i ]
#        best     = decr
#        print( "best: " + str(best) + " init exp: " + str(init_exp) )
#    elif decr < 0.5*best :
#        init_exp = init_exps[ i-1 ]
#        print( "init exp: " + str(init_exp) )
#sys.exit( 0 )
#net = torch.load( 'gn_temp.pt' )
#optimizer            = optim.Adam( net.parameters(), lr = pow(2,init_exp) )
#exp                  = init_exp
start_time           = time.time()
losses               = []
best_loss            = sys.float_info.max
last_save_epoch      = 0
last_lr_change_epoch = 0
epoch                = 0
cycle                = 1
epoch_new_cycle      = 0
# TODO add a counter and a threshold to increase the learning rate ?
while True:
    # TODO Alternative ( fastai )
    #    . multiple cycles, with a cycle corresponding to 1, k, k^2, ... epoch(s)
    #    . Decrease lr at each mini-batch, following a cosine decrease along the cycle )
    tee( '-----------------------------------------------------', logfilename )
    epoch += 1
    train_loss = 0.0
    dev_loss   = 0.0
    test_loss  = 0.0

    for _, process in enumerate( dl.Process ):
        tee( str(process), logfilename )
        gaze_dataset.setProcess( process )

        if ( dl.Process.Train == process ) :
            net = net.train()
        else :
            net = net.eval()

        loss = 0.0
        for i, data in enumerate( data_loader, 0 ):
            images         = data[ "image" ].float()
            coords         = torch.squeeze( data[ "coords" ].float(), dim=1 )
            images, coords = images.to( device ), coords.to( device )

            # zero the parameter's gradients
            optimizer.zero_grad()

            # forward, backward, update
            outputs = net( images )
            if ( i*batch_size < 4 ) or ( 0 == i*batch_size % 10000 ) :
               tee( '--- ' + str(i*batch_size) + '\n', logfilename );
               tee( 'target:\n' + str(coords), logfilename  );
               tee( 'output:\n' + str(outputs), logfilename  );

            batch_loss = criterion( outputs, coords )
            if ( dl.Process.Train == process ):
                batch_loss.backward()
                optimizer.step()

            loss = ( i*loss + batch_loss.item() ) / (i+1)

        if ( dl.Process.Train == process ) : train_loss = loss
        if ( dl.Process.Dev   == process ) : dev_loss   = loss
        if ( dl.Process.Test  == process ) : test_loss  = loss

    losses.append( train_loss )
    tee( '-------------------', logfilename )
    tee( '[%d] | cycle: %d | lr: 2^%d | train_loss: %.6f | dev_loss: %.6f | test_loss: %.6f | '
            % ( epoch, cycle, exp, train_loss, dev_loss, test_loss), logfilename )

    stop = False
    save = False

    # Save is best loss so far
    if ( dev_loss < best_loss ):
        save      = True
        best_loss = dev_loss

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
            tee( 'learning rate => 2^' + str(exp), logfilename )

    # Check if end of a cycle is reached, stop if all cycles performed, restart otherwise
    if ( exp <= -30 ) or ( train_loss < 0.0001 ):
        if cycle == n_cycles :
            stop = True
            save = True
        else :
            cycle          += 1
            epoch_new_cycle = 0
            exp             = init_exp
            update_lr( optimizer, lr=pow(2.0,exp) )
            tee( 'learning rate => 2^' + str(exp), logfilename )

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
        last_save_epoch = epoch
        torch.save( net, results_dir_name + 'gn_320x240_mse'+str(dev_loss)+'_epoch'+str(epoch)+'.pt' )
        tee( 'save net', logfilename )

    # Stop
    if stop:
        break

elapsed_time = time.time() - start_time
tee( 'Finished training in:' + str(elapsed_time), logfilename )
