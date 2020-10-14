import sys
import tables
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

input_file = sys.argv[1]
output_path = sys.argv[2]
binning = sys.argv[3]

os.system('mkdir '+output_path)

frame = pd.read_hdf(input_file)
frame_pos = frame.query('v_mu_label > 0.5')
frame_neg = frame.query('v_mu_label < 0.5')

bins = binning.split(',')

frame = frame.filter(regex='_e')

i = 1

for column in frame.columns:
    fig = plt.figure(i)

    plt.hist(frame_neg[column], np.arange(float(bins[0]), float(bins[1]), float(bins[2])), histtype='stepfilled', edgecolor=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392), linewidth=1.2, label='true negative', fc=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 0.5))
    plt.hist(frame_pos[column], np.arange(float(bins[0]), float(bins[1]), float(bins[2])), histtype='stepfilled', edgecolor=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), linewidth=1.2, label='true positive', fc=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 0.5))

    # if sample == 'trans': plt.title('Transverse polarization')
    # elif sample == 'long': plt.title('Longitudinal polarization')
    # elif sample == 'unpol': plt.title('Unpolarized OSP')
    # elif sample == 'full': plt.title('Full computation')
    # else : print('wrong polarization, skipping title')
    plt.legend(loc='upper left', ncol=2, fancybox=True, fontsize=10)
    plt.xlabel('NN score')
    plt.ylabel('Number of events')
    plt.ylim((0, 1.2*plt.ylim()[1]))

    fig.savefig(output_path + '/' + column[:-5] + '_score.pdf')
    i = i + 1
