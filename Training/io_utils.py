import os
import shutil
import datetime



logfile='log.txt'


def tee( string ) :
    print( string )
    with open( logfile, 'a' ) as f:
        f.write( string )
        f.write( '\n' )


# . Create results directory
# . Copy neural net definition file in it
# . Make the logfile point to this directory
def setup_io():
    results_dir_name = 'Results/' + datetime.datetime.now().strftime( "%Y_%m_%d-%H_%M_%S/" )
    os.makedirs( name=results_dir_name, mode=0o775, exist_ok=True )
    shutil.copy( "gaze_net.py",  results_dir_name );
    shutil.copy( "train.py",     results_dir_name );
    shutil.copy( "data_load.py", results_dir_name );
    global logfile
    logfile = results_dir_name + '/log.txt'

    return results_dir_name
