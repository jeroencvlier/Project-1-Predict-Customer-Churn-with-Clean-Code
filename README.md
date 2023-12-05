# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project aims to predict customer churn. The steps consist of importing data, exploratory data 
analysis, feature engineering and training 2 models which are Logistic Regression and Random Forest. 
For the Random Forest model, a hyperparameter tuning is performed using the GridSearchCV method. 
The best model is selected and feature importances are extracted. The two models are then saved in 
the models folder. 


## Files and data description
The project contains the following files and folders:

- `data` folder: contains the data used for the project
- `images/eda` folder: contains the images of the exploratory data analysis
- `images/results` folder: contains the images of the classification reports
- `models` folder: contains the saved models
- `logs` folder: contains the log file
- `Predict Customer Churn.ipynb`: notebook containing the origional code for the project
- `churn_library.py`: python file containing the modularised functions used in the notebook
- `churn_script_logging_and_tests.py`: python file containing the pytests for the functions in `churn_library.py`
- `conftest.py`: python file containing the fixtures for the pytests in `churn_script_logging_and_tests.py`
- `constants.py`: python file containing the constants used in the project
- `requirements.txt`: text file containing the required packages for the project


## Running Files
In order to run this project you need to have instatlled the required packages in the `requirements.txt` file
as well as have python 3.8 installed. You can install the requiremenst by running the following command in the terminal:

`pip install -r requirements.txt`

When training the model, everal images will be created in the `images/eda` folder for the exploratory data analysis.
The models will also be trained and save the models in the `models` folder. A log file is also created in the `logs` 
folder. The classification report is also generated in the `images/results` folder, along with feature importances.
To run and train the model, you can run the following command in the terminal:

`python churn_library.py`

To run the tests with pytest, you can run the following command in the terminal:

`pytest churn_script_logging_and_tests.py`



