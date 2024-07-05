# Arabic Dialects Prediction

## Overview

This repository contains code and resources for predicting Arabic dialects using machine learning models. The project includes data fetching, preprocessing, model training, and deployment scripts. You can try the Arabic Dialects Prediction app [here](https://arabic-dialects-prediction-app.streamlit.app/).

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Models](#models)
- [Usage](#usage)
- [Results and Performance](#results-and-performance)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    
bash
    git clone https://github.com/alaaashraf24/Arabic-Dialects-Prediction.git
    cd Arabic-Dialects-Prediction


2. Install the required packages:
    
bash
    pip install -r requirements.txt


## Data

The dataset used for training the models is preprocessed_dialects_data.csv. It contains text samples labeled with their corresponding dialects.

## Models

The repository includes several models:

- Logistic Regression (log_reg_model.pkl)
- Weighted Logistic Regression (log_reg_weighted_model.pkl)
- Naive Bayes (nb_model_weighted.pkl)
- Deep Learning Model (deep_learning_model.h5)

## Usage

### Data Fetching:

Run data_fetching.ipynb to fetch and prepare the dataset.

### Data Preprocessing:

Use data_preprocessing.ipynb for data cleaning and preprocessing.

### Model Training:

Execute models_training.ipynb to train the models.

### App Deployment:

Use dialectApp.py to deploy the model using Streamlit.

## Results and Performance

The models were evaluated on a test set, and the results are as follows:

### Logistic Regression:
- Accuracy: 82%
- F1-Score: 79%

### Weighted Logistic Regression:
- Accuracy: 81%
- F1-Score: 78%

### Naive Bayes:
- Accuracy: 81%
- F1-Score: 77%

### Deep Learning Model:
- Accuracy: 39%
- F1-Score: 11%

## Contributing

Contributions are welcome! Please fork the repository and create a pull request.

## License

This project is licensed under the MIT License.

