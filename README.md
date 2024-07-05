# Arabic-Dialects-Prediction
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arabic Dialects Prediction</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>Arabic Dialects Prediction</h1>
        <img src="https://yourimageurl.com" alt="Arabic Dialects" class="header-image">
        
        <h2>Overview</h2>
        <p>This repository contains code and resources for predicting Arabic dialects using machine learning models. The project includes data fetching, preprocessing, model training, and deployment scripts. You can try the Arabic Dialects Prediction app <a href="https://arabic-dialects-prediction-app.streamlit.app/" target="_blank">here</a>.</p>
        <img src="https://yourappimageurl.com" alt="App Screenshot" class="app-image">
        
        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#installation">Installation</a></li>
            <li><a href="#data">Data</a></li>
            <li><a href="#models">Models</a></li>
            <li><a href="#usage">Usage</a></li>
            <li><a href="#results-and-performance">Results and Performance</a></li>
            <li><a href="#contributing">Contributing</a></li>
            <li><a href="#license">License</a></li>
        </ul>
        
        <h2 id="installation">Installation</h2>
        <ol>
            <li>Clone the repository:
                <pre><code>git clone https://github.com/alaaashraf24/Arabic-Dialects-Prediction.git
cd Arabic-Dialects-Prediction
</code></pre>
            </li>
            <li>Install the required packages:
                <pre><code>pip install -r requirements.txt
</code></pre>
            </li>
        </ol>

        <h2 id="data">Data</h2>
        <p>The dataset used for training the models is <code>preprocessed_dialects_data.csv</code>. It contains text samples labeled with their corresponding dialects.</p>

        <h2 id="models">Models</h2>
        <p>The repository includes several models:</p>
        <ul>
            <li>Logistic Regression (<code>log_reg_model.pkl</code>)</li>
            <li>Weighted Logistic Regression (<code>log_reg_weighted_model.pkl</code>)</li>
            <li>Naive Bayes (<code>nb_model_weighted.pkl</code>)</li>
            <li>Deep Learning Model (<code>deep_learning_model.h5</code>)</li>
        </ul>

        <h2 id="usage">Usage</h2>
        <h3>Data Fetching:</h3>
        <p>Run <code>data_fetching.ipynb</code> to fetch and prepare the dataset.</p>
        <h3>Data Preprocessing:</h3>
        <p>Use <code>data_preprocessing.ipynb</code> for data cleaning and preprocessing.</p>
        <h3>Model Training:</h3>
        <p>Execute <code>models_training.ipynb</code> to train the models.</p>
        <h3>App Deployment:</h3>
        <p>Use <code>dialectApp.py</code> to deploy the model using Streamlit.</p>

        <h2 id="results-and-performance">Results and Performance</h2>
        <p>The models were evaluated on a test set, and the results are as follows:</p>
        <ul>
            <li>Logistic Regression:
                <ul>
                    <li>Accuracy: X%</li>
                    <li>F1-Score: Y%</li>
                </ul>
            </li>
            <li>Weighted Logistic Regression:
                <ul>
                    <li>Accuracy: X%</li>
                    <li>F1-Score: Y%</li>
                </ul>
            </li>
            <li>Naive Bayes:
                <ul>
                    <li>Accuracy: X%</li>
                    <li>F1-Score: Y%</li>
                </ul>
            </li>
            <li>Deep Learning Model:
                <ul>
                    <li>Accuracy: X%</li>
                    <li>F1-Score: Y%</li>
                </ul>
            </li>
        </ul>
        <p>The deep learning model showed the best performance with the highest accuracy and F1-Score.</p>

        <h2 id="contributing">Contributing</h2>
        <p>Contributions are welcome! Please fork the repository and create a pull request.</p>

        <h2 id="license">License</h2>
        <p>This project is licensed under the MIT License.</p>
    </div>
</body>
</html>
