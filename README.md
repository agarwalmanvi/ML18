## Machine Learning project 2018/19: Ensemble Fair Algorithms for Classification

### Installation set-up

* create a virtual environment with Conda
```
conda create --name aif360 python=3.5
```
To activate the environment, type
```
conda activate aif360
```
or `source activate aif360` for older version of conda\\
or `activate aif360` for Windows.\\
To deactivate the environment, use `deactivate` instead of `activate`.

* install requirements provided by `requirements.txt`.\\
This file consists of a list of packages used for this project. You can install the requirements in the virtual environment after activating it by typing the following:
```
pip install -r requirements.txt
```

### Project source codes
* `dataset`: a folder which contains the three original data sets used in this project (Adult, Compas, German)
* `results`: folder which stores all the results in csv format
* `codes`: original code for the three classifiers and ensemble. There is an output folder which stores the output data after classification

### Testing examples
* demo.ipynb : script for demo. Store results for accuracy and fairness metrics as a dataframe for all classifiers.
* demo.R : script for the plots of the results.
* alphaCalc.R : script for the alpha plot
* metrics-test.ipynb : script for testing the empirically calculated fairness metrics
