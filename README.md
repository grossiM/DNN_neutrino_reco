#  Comparing traditional and deep-learning techniques of kinematic reconstruction for polarization discrimination in vector boson scattering

### Introduction

This repository contains the code used to study the Vector Boson Scattering, in particular the neutrino polarization and W boson reference frame using classical and deep learning techniques. Please cite this work if using this code: 

@article{Grossi_2020,
   title={Comparing traditional and deep-learning techniques of kinematic reconstruction for polarization discrimination in vector boson scattering},
   volume={80},
   ISSN={1434-6052},
   url={http://dx.doi.org/10.1140/epjc/s10052-020-08713-1},
   DOI={10.1140/epjc/s10052-020-08713-1},
   number={12},
   journal={The European Physical Journal C},
   publisher={Springer Science and Business Media LLC},
   author={Grossi, M. and Novak, J. and Kerševan, B. and Rebuzzi, D.},
   year={2020},
   month={Dec}
}

https://link.springer.com/article/10.1140%2Fepjc%2Fs10052-020-08713-1

The applicability of the provided workflow goes beyond the specific use in the context of the HEP event generation. The parts of it can be reused for any simple DNN application. It provides a framework for a grid search optimization, where several DNN models are trained in a sequence, each with a different parameter combination.

### Requirements

- Python                3.7
- Root                  6.18/04
- Keras                 2.3.1
- root-numpy            4.8.0
- scikit-learn          0.23.1
- tensorflow            2.1.0
- uproot                3.11.2
- Pandas                1.2.2
- root-numpy            4.8.0

We strongly advice to create a dedicated virtual python environment using 
[conda](https://docs.conda.io/projects/conda/en/latest/index.html) or 
one of the other solutions.

### Setup
1. Clone the repository on local machine where calculation will be executed. Set the environment variable `NEW_REPO` to the directory containing `DNN_neutrino_reco`:

```
export NEW_REPO=/path/to/repository
```

## How to contribute

Contributions are welcomed as long as one sticks to the **git-flow**: fork 
this repo, create a local branch named 'feature-XXX'. Commit often. Split 
it in multiple commits and request a merge to the mainline often. 

To add new contribution please remember to follow 
[PEP 8](https://www.python.org/dev/peps/pep-0008/)
style guide, add enough comments to make the code understandable to other
user/developer and add detailed docstring following the [numpy](https://numpydoc.readthedocs.io/en/latest/format.html)
style.

## Authors

This project has been developed thanks to the effort of the following people:

- Michele Grossi
- Jakob Novak
- Matteo Marchegiani


## Usage
The flow of a data analysis like the one proposed here is the following:
- generate MC samples in the standard LHE format (i.e. PHANTOM or MADGRAPH)
- creation of root file containing physical information and different variables:

  ./write_lhe2root /path/to/generations /final/destination/path
  
- PREPROCESS of root file using python/root libraries in order to create HDF python dataset to be used in the Deep Neural Network framework. In the following we split the dataset in train, test and validation:

  python3 create_preprocessData_fullep.py -in '/path/to/rootfiles/fulcomp…..root' 
          -o '/path/to/destination/preprocessed'
          -n 'name_of_the_file'
          -s  '0.4:0.4:0.2' #separation ratio
          -ch 1 #select channel 0 semi 1 full lep
          #(create_preprocessData.py for semileptonic channel)
          
- TRAINING:  
  python3 Training.py -c /path/to/JobOption/NNconfig.cfg
  --> this will create all combination of different NN layout and produce separate naming folders where models, loss, training logs are saved

- EVALUATION: 
  python3 Evaluate.py -c /path/to/JobOption/NNconfig.cfg -p neu
  --> this will avluate all the DNN models, created in the previous step
  
- PLOT:
  python3  plot_evaluated.py -c JobOption/NNplot_config.cfg
    or
  python3 NNplot_compare.py -c JobOption/NNplot_config.cfg
    or
  python3 plot_ful_lep_all_pol.py -c JobOption/NNplot_fullylep.cfg
    or
  python3 plot_scatter.py
    or
  python3 hist_root_python_converter.py -c plt_hist.cfg -s 0(1)

