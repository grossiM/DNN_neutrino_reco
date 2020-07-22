import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def set_ticks(axis, step, integer=False, locs=None, labels=None):
	f = None
	if axis == 'x':
		f = plt.xticks
	else:
		if axis == 'y':
			f = plt.yticks
		else:
			sys.exit("'" + axis + "' axis does not exist.")

	if type(locs) is type(None):
		locs, l = f()
	ticks = np.arange(locs[0], locs[-1] + step, step)	# Include endpoint
	if type(labels) is type(None):
		labels = []
		for x in ticks:
			if x in locs:
				if integer == True:
					labels.append(int(x))
				else:
					labels.append(round(x, 1))
			else:
				labels.append(None)
	f(ticks=ticks, labels=labels)
	return

if len(sys.argv) < 2:
	sys.exit("Missing arguments: [dir]")
dir = "/data_CMS/cms/marchegiani/DNN/evaluation/" + sys.argv[1] + "/"
plot_dir = dir + "plots/"
hist_dir = plot_dir + "histograms/"
for directory in [plot_dir, hist_dir]:
	if not os.path.exists(directory):
		os.makedirs(directory)
ls = os.listdir(dir)
input_file = ""
for file in ls:
	if 'h5' in file:
		input_file = dir + file
frame = pd.read_hdf(input_file)

label_list =  ["$Z_LZ_L$", "$Z_LZ_T$", "$Z_TZ_T$"]
xlabel_list = ["$cosθ_{e^-}$", "$\eta^{Z_e}$", "$P_{T}^{Z_e}$ [GeV]", "$M_{ZZ}$ [GeV]"]
ylabel_list = ["$d\sigma/dcosθ_{e^-}$ [a.u.]", "$d\sigma/d\eta^{Z_e}$ [a.u.]", "$d\sigma/P_{T}^{Z_e}$ [a.u.]", "$d\sigma/dM_{ZZ}$ [a.u.]"]
legend_list = ["center", "upper right", "upper right", "upper right"]
xstep_list =  [0.1, 0.5, 20, 50]
integer_list =[False, True, True, True]
locs_list =   [np.arange(-1, +1.2, 0.2), None, None, None]
xticks_list = [None, None, None, None]

categories = ['LL', 'LT', 'TT']
score_labels = ['LL score', 'LT score', 'TT score']
scores = [column for column in frame.columns if 'cat' in column]
models = [score.split('_')[0] for score in scores]
models = list(dict.fromkeys(models))

frame_LL = frame[frame['longitudinal'] == 1]
frame_LT = frame[frame['mixed'] == 1]
frame_TT = frame[frame['transverse'] == 1]
colors = ['red', 'green', 'blue']

for model in models:
	vars = [score for score in scores if model in score and not "rounded" in score]
	rounded_scores = [score for score in scores if model in score and "rounded" in score]
	pred_LL = frame[frame[rounded_scores[0]] == 1]
	pred_LT = frame[frame[rounded_scores[1]] == 1]
	pred_TT = frame[frame[rounded_scores[2]] == 1]
	plt.figure(figsize=[12, 10])
	for (i, var) in enumerate(vars):
		ax = plt.subplot(2,2,i+1)
		for (j, df) in enumerate([frame_LL, frame_LT, frame_TT]):
			plt.hist(df[var].values, bins=np.linspace(0,1,26), histtype='step', ec=colors[j], label=categories[j])
			handles, labels = ax.get_legend_handles_labels()
			new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
			#plt.legend(loc='best')
		set_ticks('x', 0.1, locs=np.linspace(0, 1.0, 11))
		plt.xlim(0.0, 1.0)
		plt.xlabel(score_labels[i])
		plt.ylabel("[a.u.]")
		plt.title(categories[i] + ' score')
	plt.subplot(2,2,4)
	#plt.ylim(0,1)
	plt.legend(handles=new_handles, labels=labels, loc=legend_list[i], fontsize=15, bbox_to_anchor=(0.6, 0.6), frameon=True)
	file = plot_dir + model+ "_pred.png"
	print("Saving " + file)
	plt.savefig(file, format="png")
	plt.close()
	vars = ["Theta_e", "ze_eta", "ze_pt", "m4l"]
	plt.figure(figsize=[12, 10])
	bins_list = [np.linspace(-1,1,41), np.linspace(-6, 6, 49), np.linspace(0, 600, 61), np.linspace(200, 1200, 101)]
	varnames = ["costheta", "eta_z", "pt_z", "mzz"]
	for (i, var) in enumerate(vars):
		n_list = []
		ax = plt.subplot(2,2,i+1)
		for (j, df) in enumerate([pred_LL, pred_LT, pred_TT]):
			if varnames[i] == "costheta":
				n, b, patches = plt.hist(np.cos(df[var].values), bins=bins_list[i], histtype='step', ec=colors[j], label=categories[j])
			else:
				n, b, patches = plt.hist(df[var].values, bins=bins_list[i], histtype='step', ec=colors[j], label=categories[j])
			n_list.append(n)
		sigma_sum = np.zeros_like(n_list[0])
		for n in n_list:
			sigma_sum += n
		bins = bins_list[i]
		bins_center = [x + 0.5*(bins[1]-bins[0]) for x in bins[:-1]]
		n, b, patches = plt.hist(bins_center, bins=bins, weights=sigma_sum, color="purple", linewidth=1, label="$\Sigma$ pol.", histtype="step", ec="purple")

		plt.xlim(b[0], b[-1])
		ylim = 1.05*max(n)
		ybbox = 0.65
		if var != "m4l":
			plt.ylim(0, ylim)
		handles, labels = ax.get_legend_handles_labels()
		new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
		plt.legend(handles=new_handles, labels=labels, loc=legend_list[i], fontsize=12, frameon=False)
		if var == "Theta_e":
			plt.legend(handles=new_handles, labels=labels, loc=legend_list[i], fontsize=12, bbox_to_anchor=(0.5, ybbox), frameon=False)

		xlabel = xlabel_list[i]
		ylabel = ylabel_list[i]
		set_ticks('x', xstep_list[i], integer=integer_list[i], locs=locs_list[i], labels=xticks_list[i])
		plt.xlim(b[0], b[-1])
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(ylabel)
		if var == "m4l":
			plt.yscale('log')
	file = hist_dir + model+ "_vars.png"
	print("Saving " + file)
	plt.savefig(file, format="png")
	plt.close()

for model in models:
	vars = [score for score in scores if model in score and not "rounded" in score]
	rounded_scores = [score for score in scores if model in score and "rounded" in score]
	pred_LL = frame[frame[rounded_scores[0]] == 1]
	pred_LL = pred_LL[pred_LL[rounded_scores[1]] == 0]
	pred_LL = pred_LL[pred_LL[rounded_scores[2]] == 0]
	#pred_LL = pred_LL[pred_LL[vars[1]] < pred_LL[vars[0]]]
	#pred_LL = pred_LL[pred_LL[vars[2]] < pred_LL[vars[0]]]
	eff_LL = float(pred_LL.shape[0])/float(frame_LL.shape[0])
	pred_LT = frame[frame[rounded_scores[1]] == 1]
	pred_LT = pred_LT[pred_LT[rounded_scores[0]] == 0]
	pred_LT = pred_LT[pred_LT[rounded_scores[2]] == 0]
	#pred_LT = pred_LT[pred_LT[vars[0]] < pred_LT[vars[1]]]
	#pred_LT = pred_LT[pred_LT[vars[2]] < pred_LT[vars[1]]]
	eff_LT = float(pred_LT.shape[0])/float(frame_LT.shape[0])
	pred_TT = frame[frame[rounded_scores[2]] == 1]
	pred_TT = pred_TT[pred_TT[rounded_scores[0]] == 0]
	pred_TT = pred_TT[pred_TT[rounded_scores[1]] == 0]
	#pred_TT = pred_TT[pred_TT[vars[0]] < pred_TT[vars[2]]]
	#pred_TT = pred_TT[pred_TT[vars[1]] < pred_TT[vars[2]]]
	eff_TT = float(pred_TT.shape[0])/float(frame_TT.shape[0])
	eff = [eff_LL, eff_LT, eff_TT]
	print(eff)
	plt.figure(figsize=[12, 10])
	for (i, var) in enumerate(vars):
		ax = plt.subplot(2,2,i+1)
		for (j, df) in enumerate([pred_LL, pred_LT, pred_TT]):
			plt.hist(df[var].values, bins=np.linspace(0,1,26), histtype='step', ec=colors[j], label=categories[j])
			handles, labels = ax.get_legend_handles_labels()
			new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
			#plt.legend(loc='best')
		set_ticks('x', 0.1, locs=np.linspace(0, 1.0, 11))
		plt.xlim(0.0, 1.0)
		plt.xlabel(score_labels[i])
		plt.title(categories[i] + ' score')
	plt.subplot(2,2,4)
	#plt.ylim(0,1)
	plt.legend(handles=new_handles, labels=labels, loc=legend_list[i], fontsize=15, bbox_to_anchor=(0.6, 0.6), frameon=True)
	file = plot_dir + model+ "_pred_no2cats.png"
	print("Saving " + file)
	plt.savefig(file, format="png")
	plt.close()
	vars = ["Theta_e", "ze_eta", "ze_pt", "m4l"]
	plt.figure(figsize=[12, 10])
	bins_list = [np.linspace(-1,1,41), np.linspace(-6, 6, 49), np.linspace(0, 600, 61), np.linspace(200, 1200, 101)]
	varnames = ["costheta", "eta_z", "pt_z", "mzz"]
	for (i, var) in enumerate(vars):
		n_list = []
		ax = plt.subplot(2,2,i+1)
		for (j, df) in enumerate([pred_LL, pred_LT, pred_TT]):
			if varnames[i] == "costheta":
				m, b = np.histogram(np.cos(df[var].values), bins=bins_list[i])
				step = b[1] - b[0]
				bins_center = [x + 0.5*step for x in b[:-1]]
				#n, b, patches = plt.hist(bins_center, bins=bins_list[i], weights=(1./eff[j])*np.array(m), histtype='step', ec=colors[j], label=categories[j])
				n, b, patches = plt.hist(bins_center, bins=bins_list[i], weights=np.array(m), histtype='step', ec=colors[j], label=categories[j])
			else:
				m, b = np.histogram(df[var].values, bins=bins_list[i])
				step = b[1] - b[0]
				bins_center = [x + 0.5*step for x in b[:-1]]
				#n, b, patches = plt.hist(bins_center, bins=bins_list[i], weights=(1./eff[j])*np.array(m), histtype='step', ec=colors[j], label=categories[j])
				n, b, patches = plt.hist(bins_center, bins=bins_list[i], weights=np.array(m), histtype='step', ec=colors[j], label=categories[j])
			n_list.append(n)
		sigma_sum = np.zeros_like(n_list[0])
		for n in n_list:
			sigma_sum += n
		bins = bins_list[i]
		bins_center = [x + 0.5*(bins[1]-bins[0]) for x in bins[:-1]]
		n, b, patches = plt.hist(bins_center, bins=bins, weights=sigma_sum, color="purple", linewidth=1, label="$\Sigma$ pol.", histtype="step", ec="purple")
		
		plt.xlim(b[0], b[-1])
		ylim = 1.05*max(n)
		ybbox = 0.65
		if var != "m4l":
			plt.ylim(0, ylim)
		handles, labels = ax.get_legend_handles_labels()
		new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
		plt.legend(handles=new_handles, labels=labels, loc=legend_list[i], fontsize=12, frameon=False)
		if var == "Theta_e":
			plt.legend(handles=new_handles, labels=labels, loc=legend_list[i], fontsize=12, bbox_to_anchor=(0.5, ybbox), frameon=False)

		xlabel = xlabel_list[i]
		ylabel = ylabel_list[i]
		set_ticks('x', xstep_list[i], integer=integer_list[i], locs=locs_list[i], labels=xticks_list[i])
		plt.xlim(b[0], b[-1])
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(ylabel)
		if var == "m4l":
			plt.yscale('log')
	file = hist_dir + model+ "_vars_no2cats.png"
	print("Saving " + file)
	plt.savefig(file, format="png")
	plt.close()