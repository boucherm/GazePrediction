import sys
import time
import configparser
import math
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
import io_utils  as iu



# Visualization tool
#https://www.wandb.com/blog/monitor-your-pytorch-models-with-five-extra-lines-of-code

# Update the learning rate of an optimizer
# TODO put in a train_utils module
def update_lr( optimizer, lr ) :
    #print( optimizer.state_dict()['param_groups'] )
    #print( optimizer.state_dict()['param_groups'][0]['lr'] )
    optimizer_state =  optimizer.state_dict()
    optimizer_state['param_groups'][0]['lr'] = lr
    optimizer.load_state_dict( optimizer_state )
    #optimizer.state_dict()['param_groups'][0]['lr'] = 2
    #print( optimizer.state_dict()['param_groups'][0]['lr'] )
    #print( optimizer.state_dict()['param_groups'] )




# Hyper-parameters
init_exp = -8 # TODO: a first pass to search for a good value of init_exp
n_cycles = 9
N        = 8 # batch_size, beware, for '1' batch_norm will fail and raise an exceptijn

results_dir_name = iu.setup_io()

config = configparser.ConfigParser()
config.read( '../config.txt' )
screen_w = int( config['SCREEN']['width'] )
screen_h = int( config['SCREEN']['height'] )
img_w    = int( config['IMAGE']['width'] )
img_h    = int( config['IMAGE']['height'] )

# Setting fixed seed, for debug purposes
#torch.manual_seed(0)
# Alternative way to set workers' seeds, pass the following fn as worker_init_fn
#def _init_fn(worker_id):
#    np.random.seed(12 + worker_id)

iu.tee( "Preparing dataset" )
#data_channels = dl.DataChannels.One
#data_channels = dl.DataChannels.All
gaze_dataset = dl.GazeDataset( "../Data",
                               target_res=(img_w,img_h),
                               transform=transforms.Compose( [
                                   dl.ScaleImage( (img_w,img_h) ),
                                   dl.NormalizeCoordinates( (screen_w,screen_h) ),
                                   dl.CenterPixels(),
                                   dl.ToTensor() ] ) )
iu.tee( "done" )

iu.tee( "Preparing dataloader" )
data_loader = DataLoader( gaze_dataset, batch_size=N, shuffle=True, num_workers=3, drop_last=True )
iu.tee( "done" )

iu.tee( "Loading net" )
#net = gn.GazeNet320x240_Strided()
net = gn.GazeNetProgressive( img_w, img_h,
                             [ gn.CP(7,2,64),
                               gn.CP(7,2,16),
                               gn.CP(7,2,16),
                               gn.CP(5,2,16),
                               gn.CP(5,1,16),
                               gn.CP(5,1,16) ],
                             [ gn.FCP( gn.AOR.Abs, 512 ),
                               gn.FCP( gn.AOR.Abs, 512 ),
                               gn.FCP( gn.AOR.Abs, 512 ),
                               gn.FCP( gn.AOR.Abs, 512 ),
                               gn.FCP( gn.AOR.Abs, 2 ) ],
                             #True, N
                             False, N
                           )
#print( net )
#net = torch.load ('Results/2018_12_18-00_56_43/gn_320x240_mse0.17132242421309155_epoch95.pt' )
device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
#device = "cpu"
net.to( device )
iu.tee( "done" )
iu.tee( "device: " + str(device) )

with open( 'continue.txt', 'w' ) as f :
    f.write( "1" )
f.close()

iu.tee( 'begin training' )
criterion = nn.MSELoss()
#criterion = nn.L1Loss()
exp       = init_exp
optimizer = optim.Adam( net.parameters(), lr = pow(2,exp) )

#import IPython
#IPython.core.debugger.set_trace()
#sys.exit( 0 )

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
best_train_loss      = sys.float_info.max
best_dev_loss        = sys.float_info.max
last_save_epoch      = 0
last_dropout_epoch   = -1
last_lr_change_epoch = 0
epoch                = 0
cycle                = 1
epoch_new_cycle      = 0
frozen_convs         = False
# TODO add a counter and a threshold to increase the learning rate ?
while True:
    # Alternatives ( fastai )
    #    . multiple cycles, with a cycle corresponding to 1, k, k^2, ... epoch(s)
    #    . Decrease lr at each mini-batch, following a cosine decrease along the cycle )
    iu.tee( '-----------------------------------------------------' )
    epoch     += 1
    train_loss = 0.0
    dev_loss   = 0.0
    test_loss  = 0.0

    for _, process in enumerate( dl.Process ):
        iu.tee( str(process) )
        gaze_dataset.setProcess( process )

        if ( dl.Process.Train == process ) :
            net = net.train()
        else :
            net = net.eval()

        loss = 0.0
        start = time.time()
        for i, data in enumerate( data_loader, 0 ):
            images         = data[ "image" ].float()
            coords         = torch.squeeze( data[ "coords" ].float(), dim=1 )
            images, coords = images.to( device ), coords.to( device )

            # zero the parameter's gradients
            optimizer.zero_grad()

            # forward
            outputs = net( images )
            if ( i == 0 ) or ( i == math.floor( len(gaze_dataset)/(2*N) ) ):
               print()
               iu.tee( '--- ' + str(i*N) + '\n' )
               iu.tee( 'target:\n' + str(coords[0:4,:]) )
               iu.tee( 'output:\n' + str(outputs[0:4,:]) )

            #if 1000 == i :
                #end = time.time()
                #print()
                #print( end - start )
                #sys.exit(0)

            # backward & update
            batch_loss = criterion( outputs, coords )
            if ( dl.Process.Train == process ):
                batch_loss.backward()
                optimizer.step()
                #if ( hasattr( net, 'norm_abs' ) ) :
                    #net.norm_abs()

            loss = ( i*loss + batch_loss.item() ) / (i+1)
            print( process, ' ', i, ' - ', batch_loss.item(), ' - ', loss, ' - ',
                   math.sqrt(loss/4)*( screen_w+screen_h )/2, end='\r' )

        print()
        if ( dl.Process.Train == process ) : train_loss = loss
        if ( dl.Process.Dev   == process ) : dev_loss   = loss
        if ( dl.Process.Test  == process ) : test_loss  = loss

    losses.append( train_loss )
    iu.tee( '-------------------' )
    iu.tee( '[%d] | cycle: %d | lr: 2^%d | train_loss: %.6f | dev_loss: %.6f | test_loss: %.6f | '
            % ( epoch, cycle, exp, train_loss, dev_loss, test_loss) )

    stop = False
    save = False

    #--- Save best losses
    mse_str = ''
    if ( dev_loss < best_dev_loss ):
        save          = True
        best_dev_loss = dev_loss
        frac, inte    = math.modf( dev_loss )
        mse_str       = '_dev' + '_mse' + str(int(inte)) + '_' + str(frac)[2:6]

    if ( train_loss < 0.8*best_train_loss ): # otherwise we'll be saving way too often
        save            = True
        best_train_loss = train_loss
        frac, inte      = math.modf( train_loss )
        mse_str         = '_train' + '_mse' + str(int(inte)) + '_' + str(frac)[2:6]

    #--- Check if lr reduction is needed, perform if so
    e = epoch - 1
    if ( ( epoch - epoch_new_cycle ) >= 2 ) :
        do_update_lr = ( ( losses[e] > 0.97*losses[e-1] ) and ( ( epoch - last_dropout_epoch ) > 1 ) )
        if do_update_lr :
            delta_lr_change      = ( epoch - last_lr_change_epoch )
            exp                 -= 1 if ( delta_lr_change > 2 ) else 2
            last_lr_change_epoch = epoch
            update_lr( optimizer, lr=pow(2.0,exp) )
            iu.tee( 'learning rate => 2^' + str(exp) )

    #--- Increase dropout probabilities in case of overfitting
    if ( train_loss < 0.9*dev_loss ):
        if ( hasattr( net, 'increase_dropouts' ) ) :
            if net.increase_dropouts() :
                last_dropout_epoch = epoch

    #--- Check if end of a cycle is reached, stop if all cycles performed, restart otherwise
    if ( exp < -30 ) or ( dev_loss < 0.0001 ):
        if cycle == n_cycles :
            stop = True
            save = True
        else :
            iu.tee( '-------------------' )
            iu.tee( 'New cycle' )
            cycle          += 1
            epoch_new_cycle = epoch
            exp             = init_exp
            if hasattr( net, 'capacity' ) and hasattr( net, 'freeze_convs' ):
                if frozen_convs:
                    frozen_convs = False
                    net.freeze_convs( False )
                elif ( net.capacity() < net.max_capacity() ):
                    net.freeze_convs()
                    frozen_convs = True
                    net.increase_capacity( 1 )
                exp = ( init_exp ) if (frozen_convs) else ( init_exp - net.capacity() - 1 )
                iu.tee( 'capacity: ' + str(net.capacity()) + ', frozen convs: ' + str(frozen_convs) )
            if hasattr( net, 'reset_dropouts' ):
                net.reset_dropouts()
            iu.tee( 'learning rate => 2^' + str(exp) )
            update_lr( optimizer, lr=pow(2.0,exp) )

    #--- Read file to check if stop has been requested
    with open( 'continue.txt', 'r' ) as f :
        # XXX ^ Not robust to simultaneous read/write, those should be exceptional though
        l    = f.readline();
        stop, save = ( True, True ) if ( "1" != l ) else ( stop, save )
    f.close()

    #--- Save parameters and neural net file
    if save:
        # TODO
        #   . https://pytorch.org/docs/stable/notes/serialization.html
        #   . save optimizer state ?
        #     -> https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3
        last_save_epoch = epoch
        torch.save( net, results_dir_name + 'gn_' + 'epoch'+ str(epoch) + mse_str + '.pt' )
        iu.tee( 'save net' )

    # Stop
    if stop:
        break

elapsed_time = time.time() - start_time
iu.tee( 'Finished training in:' + str(elapsed_time) )
