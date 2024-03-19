# BF3_Affinity
Machine Learning (ML) and Deep Learning (DL) models for predicting experimental Lewis Base Affinity and Lewis Polybase Binding Atoms in the complex formation with Boron Trifluoride (BF3).

Each model includes the datasets formatted in the appropriate form, source code for training and testing processes, as well as the results of the model performance. The ML models contain the tabular data of the datasets, while the DL model (i.e., GNN) contains the graph objects of the investigated molecules.

This study aims to resolve two critical tasks in the realm of Lewis acid-base interactions that have not been quantitatively investigated. First, machine learning developed from the simulation is successful at predicting the experimental properties of Lewis adducts. Second, the models can identify the binding atoms of Lewis polybases in 1:1 adducts, which is either improbable to obtain with experiments, or computationally expensive for quantum simulation.

The publication associated with this study may be found in the JCC paper:

Huynh, H.; Le, K.; Vu, L.; Nguyen, T.; Holcomb, M.; Forli, S.; Phan, H. Synergy of Machine Learning and Density Functional Theory Calculations for Predicting Experimental Lewis Base Affinity and Lewis Polybase Binding Atoms. Journal of Computational Chemistry. [https://doi.org/10.1002/jcc.27329](https://doi.org/10.1002/jcc.27329).

If using the models in your work, please cite this publication.

# README Outline
- [ML_model](https://github.com/Comistrylab/BF3_Affinity#ml_model)
- [GNN_model](https://github.com/Comistrylab/BF3_Affinity#gnn_model)
## ML_model
### Machine Learning Models
- Linear Regression (LR)
- Ridge Regression (Ridge)
- Random Forest (RF)
- Gradient Boosting (GB)

### Dependencies
- Python 3 (download [here](https://www.python.org/downloads/))
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), and [Scipy](https://scipy.org/)

### Package Organization
This package includes the following major files:
- `main.py`: the main file to execute the program.
- `models.py`: contains all setup for ML models for each dataset.
- `load_data.py`: gets and processes datasets
- `train_model.py`: setup program to train/evaluate and cross-validate models
- `plot.py`: includes functions for plotting purposes.
- `data` folder: contains all CSV files of data
- `results` folder: contains results produced by the program

### Instruction
To obtain the results, simply execute `main.py`. The program will prompt you for the program from which you want to execute. Nothing will be printed on the screen by the program. After the program terminates, look in the `results` folder for the output.

For more program documentation, please refer to the description inside each code file and function.

## GNN_model
### Deep Learning Model
- Graph-based Neural Network (GNN)

### Dependencies
- Python 3 (download [here](https://www.python.org/downloads/))
- [Deepchem](https://deepchem.readthedocs.io/en/latest/)
- [Tensorflow](https://www.tensorflow.org/), [Pytorch](https://pytorch.org/docs/stable/), and [Scikit-learn](https://scikit-learn.org/stable/)
- [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), and [Scipy](https://scipy.org/)

### Package Organization
This package includes the following major files:
- `main.py`: the main file to execute the program including programs to train/evaluate the model.
- `plot.py`: includes functions for plotting purposes.
- `Data` folder: contains all input files of data
- `Result` folder: contains results produced by the program

### Instruction
To obtain the results, simply execute `main.py`. The program will be conducted automatically. Nothing will be printed on the screen by the program. After the program terminates, look in the `results` folder for the output.

For more program documentation, please refer to the description inside each code file and function.
