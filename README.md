# SRNN
# Language: Python
# Input: TXT
# Output: PREFIX
# Tested with: PluMA 1.1, Python 3.6

PluMA plugin that runs Staged Recurrent Neural Network (Dupond, 2019)

The plugin expectes as input a tab-delimited file of keyword-value pairs:
inputfile
hours
features
K
stagefile
nonstagefile
activation
regression
lr
epochs
slice

Output CSV files for predictions (Y and Yhat) are produced using the user-specified prefix.
