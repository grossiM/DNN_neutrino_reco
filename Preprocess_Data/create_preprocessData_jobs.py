import sys
import os
import configparser

'''

A wrapper for creating jobs for training different hyperparametrization configurations on a specific unpolarized dataset. Jobs are written in a submit_jobs.sh file. By setting flags -ex and -otf to True, it enables also running created jobs, postprocessing and plotting. It assumes existance of the "training area", containg the following four directories: 'data', 'models', 'results', 'plots'. The input data has to be stored in the 'data' directory, the rest of the folders are used for storing the outputs.
usage: python3 create_preprocessData_jobs.py JobOptions/NNconfig_pre.cfg
'''
#check path as it is not correct
config = configparser.ConfigParser()
config.optionxform = str  # to preserve the case when reading options

if len(sys.argv) < 2: 
    print('Config file missing, exiting')
    sys.exit(1)
config.read(sys.argv[1])

output_folder = config.get('general', 'output-folder')
data_area = config.get('general', 'data-area')


if not os.path.exists(data_area +'/data/rootfiles/' + output_folder):
    os.system('mkdir ' + data_area +'/data/rootfiles/' + output_folder)

submit_file = data_area +'/data/rootfiles/' + output_folder + '/submit_jobs.sh'
#submit_file = data_area + '/models/' + output_folder + '/submit_jobs.sh'

print(submit_file)
f = open(submit_file,'w')

root_set = config.get('general', 'root-set')
root_input = root_set + '.root'

#####method########
try: number_of_events = config.get('submission','number-of-events')
except: number_of_events = -1

try: channel = config.get('general','channel')
except: "ERROR: fix channel type (semilep or full lep)"
print('channel:' + channel)

try: separation = config.get('submission','separation')
except: "no sample separation selected"
print('separation:'+ separation)


if (int(config.get('general','fulleptonic'))==1) &  (int(config.get('general','semileptonic'))==1):
    print(' ERROR select only one topology decay ')
    sys.exit(1)
elif (int(config.get('general','fulleptonic'))==0) &  (int(config.get('general','semileptonic'))==0):
    print(' ERROR select one topology decay ')
    sys.exit(1)
else: 
    command = 'python3 create_preprocessData.py -in \'' + data_area + '/data/rootfiles/' + root_input + \
    '\' -o \'' + data_area + '/data/rootfiles/' + output_folder + \
    '\' -n \'' + root_set + \
    '\' -nev \'' + str(number_of_events) + \
    '\' -pol ' + channel + \
    ' -s ' + separation + '\n'
    f.write(command)
    print(command)

f.close()
os.system('chmod u+x ' + submit_file)

os.system(submit_file)