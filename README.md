# iwcd-mlselection

This repository contains the code required to analyze data produced with the [nuPRISM Analysis repo clone]{https://github.com/AntoineMsy/Analysis} code. Combining the outputs of FitQun and a CNN-based classifier, an analysis of $\nu_e$ event selection is made. Two approaches are proposed, and can be found in the notebooks : 

- `dataset_eda.ipynb` contains basic statistics of the data, then implements a method to replace FitQun cuts with some of the classifier output variables. We compare the results of two different methods.
- `sigbg_ml.ipynb` implements a Gradient Boosted Decision Tree (GBDT) using scikit-learn to determine the cuts. The results are compared with FitQun, and a qualitative analysis of the behavior of the GBDT is proposed. 

`utils.py` : contains some functions
