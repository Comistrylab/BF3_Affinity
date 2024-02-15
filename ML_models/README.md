# Machine Learning Approach for Predicting Optoelectronic Properties of LA-LB Adducts

This repository includes following major files:

- `main.py`: the main file to execute the program.
- `models.py`: contains all setup for ML models for each dataset.
- `load_data.py`: gets and processes datasets
- `train_model.py`: setup program to train/evaluate and cross validate models
- `data` folder: contains all CSV files of data
- `results` folder: contains results produced by the program

## Requirements to execute the program

- Python 3 (download [here](https://www.python.org/downloads/))
- [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), and [Scikit-learn](https://scikit-learn.org/stable/) (can use `pip` to install)

## How to use

To obtain the results, simply execute `main.py`. The program will prompt you for the program from which you want to execute. Nothing will be printed on the screen by the program. After the program terminates, look in the `results` folder for the output.

For more program's documentation, please refer to description inside each code file and function.

**Note:**

- Due to the setting of random seeds in Scikit-learn's models, the final numbers may differ slightly from those in the published paper. However, it does not affect the conclusion.
