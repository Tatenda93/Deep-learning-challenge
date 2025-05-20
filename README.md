# Alphabet Soup Charity Deep Learning Challenge

## Overview

This project involves building and optimizing a binary classification model using deep learning to predict whether applicants funded by the nonprofit Alphabet Soup will be successful. The dataset consists of over 34,000 historical applications, and the goal is to use the features provided to predict the target variable `IS_SUCCESSFUL`.

## Files Included

- `AlphabetSoupCharity.ipynb`: Initial neural network training and evaluation.
- `AlphabetSoupCharity_Optimization.ipynb`: Optimized model with improved architecture and performance.
- `AlphabetSoupCharity.h5`: Saved model from the initial run.
- `AlphabetSoupCharity_Optimization.h5`: Saved model from the optimized run.
- `AlphabetSoup_Report.docx`: Summary report of the model building and performance.
- `README.md`: Project overview and file descriptions.

## Technologies Used

- Python
- Pandas, NumPy
- TensorFlow, Keras
- Scikit-learn
- Google Colab / Jupyter Notebooks

## Model Overview

- Input Features: One-hot encoded categorical and numerical variables.
- Target: `IS_SUCCESSFUL` (1 = successful, 0 = unsuccessful).
- Architecture: Sequential model with 2 hidden layers (80 and 30 neurons), ReLU activations, sigmoid output.
- Optimizer: Adam
- Loss Function: Binary crossentropy
- Achieved Accuracy: ~75.1%

## Optimization Techniques

- Increased number of neurons
- Added more hidden layers
- Tuned epochs and batch size
- Grouped rare categorical values into “Other”
- Used callbacks for early stopping and model checkpointing

## Recommendation

For future improvements, consider testing alternative classification models such as Random Forest or XGBoost, using automated hyperparameter tuning tools, and enhancing feature engineering.
