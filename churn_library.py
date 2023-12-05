# library doc string
"""This contains functions for the churn model to perform
feature engineering, model training, and model evaluation.

Author: Jeroen van Lier

Date: 2023-12-05
"""
import os
import time
import logging
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report  # , plot_roc_curve
from constants import (
    DATA_FILE_PATH, EDA_IMAGE_PATH,
    RESULT_IMAGE_PATH, MODEL_PATH,
    keep_cols, cat_columns, param_grid
)

# set presets
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()

# create logging file
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''

    try:
        assert os.path.isfile(
            pth) is True, "FileNotFoundError - The file path is incorrect"
    except AssertionError as err:
        logging.error("FileNotFoundError - The file path is incorrect")
        raise err
    try:
        df = pd.read_csv(pth)
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        logging.info("File imported successfully")
        return df
    except Exception as err:
        logging.error("File import FAILED - %s", err)
        raise err


class Plotter:
    """
    A class for creating and saving various types of plots using Matplotlib and Seaborn.

    Attributes:
        figsize (tuple): A tuple specifying the width and height of the plots.
    """

    def __init__(self, figsize=(20, 10)):
        """
        Constructs all the necessary attributes for the Plotter object.

        Args:
            figsize (tuple, optional): A tuple specifying the width and height of the plots.
            Defaults to (20, 10).
        """
        self.figsize = figsize

    def save_and_log_plot(self, plot_name, kind, save_path):
        """
        Saves the current plot to a file and logs this event.

        Args:
            plot_name (str): The name of the plot (typically the series name).
            kind (str): The type of the plot (e.g., 'bar', 'heatmap').
            save_path (str): The file path to save the plot.
        """
        plt.savefig(save_path)
        plt.close()
        logging.info(
            '%s plot for %s saved at %s',
            kind.capitalize(),
            plot_name,
            save_path)

    def bar_plot(self, series):
        """
        Generates and saves a bar plot for the given series.

        Args:
            series (pandas.Series): The pandas Series to plot.
        """
        plt.figure(figsize=self.figsize)
        file_string = f'{EDA_IMAGE_PATH}barplot_{series.name}.png'
        plt.bar(series.index, series.values)
        self.save_and_log_plot(series.name, 'bar', file_string)

    def hist_plot(self, series):
        """
        Generates and saves a histogram for the given series.

        Args:
            series (pandas.Series): The pandas Series for which to plot the histogram.
        """
        plt.figure(figsize=self.figsize)
        file_string = f'{EDA_IMAGE_PATH}histplot_{series.name}.png'
        plt.hist(series)
        self.save_and_log_plot(series.name, 'hist', file_string)

    def snshist_plot(self, series):
        """
        Generates and saves a Seaborn histogram with kernel density estimation for the given series.

        Args:
            series (pandas.Series): The pandas Series for which to plot the Seaborn histogram.
        """
        plt.figure(figsize=self.figsize)
        file_string = f'{EDA_IMAGE_PATH}snshistplot_{series.name}.png'
        sns.histplot(series, stat='density', kde=True)
        self.save_and_log_plot(series.name, 'snshist', file_string)

    def heatmap_plot(self, corr_df):
        """
        Generates and saves a heatmap of the correlation matrix for the numeric columns of the
        given DataFrame.

        Args:
            df (pandas.DataFrame): The pandas DataFrame for which to plot the heatmap.
        """
        plt.figure(figsize=self.figsize)
        numeric_df = corr_df.select_dtypes(include=[np.number])
        file_string = f'{EDA_IMAGE_PATH}heatmap_corr.png'
        sns.heatmap(
            numeric_df.corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2)
        self.save_and_log_plot('correlation_matrix', 'heatmap', file_string)


def perform_eda(eda_df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            df: pandas dataframe (churn column modified)
    '''
    eda_df['Churn'] = eda_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    eda_df.drop(columns=['Attrition_Flag'], inplace=True)

    plotter = Plotter()
    plotter.hist_plot(eda_df['Churn'])
    plotter.hist_plot(eda_df['Customer_Age'])
    plotter.bar_plot(eda_df['Marital_Status'].value_counts('normalize'))
    plotter.snshist_plot(eda_df['Total_Trans_Ct'])
    plotter.heatmap_plot(eda_df)

    logging.info('Completed all EDA plots')

    return eda_df


def encoder_helper(encode_df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
            naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    for category in category_lst + [response]:
        assert category in encode_df.columns, f"{category} column is not in the dataframe"

    for category in category_lst:
        category_groups = encode_df.groupby(category)[response].mean()
        category_list = [category_groups[cat] for cat in encode_df[category]]
        encode_df[f'{category}_Churn'] = category_list

    return encode_df


def perform_feature_engineering(df_eng, response='Churn'):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for
              naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # create X and y
    try:
        y = df_eng[response]
    except KeyError as err:
        logging.error(
            "KeyError - The y column is not present in the dataframe, clumns available are: %s",
            df_eng.columns)
        raise err

    try:
        x = df_eng[keep_cols]
    except KeyError as err:
        missing_cols = [col for col in keep_cols if col not in df_eng.columns]
        logging.error(
            "KeyError - The keep_cols list is incorrect, missing columns are: %s",
            missing_cols)
        raise err

    x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
        x, y, test_size=0.3, random_state=42)
    logging.info("Train test split completed successfully")
    logging.info("x_train shape: %s , y_train shape: %s",
                 x_train_split.shape, y_train_split.shape)
    logging.info(
        "x_test shape: %s , y_test shape: %s",
        x_test_split.shape,
        y_test_split.shape)

    return x_train_split, x_test_split, y_train_split, y_test_split


def classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds,
                                model_name):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
            model_name: name of model used to generate predictions

    output:
             None
    '''
    logging.info("Generating classification report for %s", model_name)
    try:
        plt.rc('figure', figsize=(6, 5))
        plt.text(0.01, 1.25, str(f'{model_name.replace("_"," ")} Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    y_test, y_test_preds)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str(f'{model_name.replace("_"," ")} Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    y_train, y_train_preds)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(
            f'{RESULT_IMAGE_PATH}Classification_Report_{model_name.replace(" ","_")}.png',
            bbox_inches='tight')
        plt.close()
        logging.info(
            "Classification report for %s saved successfully",
            model_name)
    except Exception as err:
        logging.error(
            "Error generating classification report for %s",
            model_name)
        raise err


def feature_importance_plot(model, x_data):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values

    output:
             None
    '''
    # tree explainer plot
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model.best_estimator_)
    # Calculate Shap values
    shap_values = explainer.shap_values(x_data)
    # Plot summary_plot
    shap.summary_plot(shap_values, x_data, plot_type="bar", show=False)
    # save figure
    plt.savefig(
        f'{RESULT_IMAGE_PATH}Importance_Plot_tree_explainer.png',
        bbox_inches='tight')

    # feature importance
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    # save figure
    plt.savefig(
        f'{RESULT_IMAGE_PATH}Importance_Plot_feature_importance.png',
        bbox_inches='tight')

    logging.info("Feature importance plots saved successfully")


def roc_auc_plot(model, x_te, y_te, name=None):
    '''
    creates and stores the roc auc plot in pth
    input:
            model: model object
            x_te: pandas dataframe of X testing values
            y_te: pandas dataframe of y testing values

    output:
             None
    '''
    logging.info("Generating ROC AUC plot")
    # Compute the probabilities
    y_score = model.predict_proba(x_te)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_te, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {name}')
    plt.legend(loc="lower right")
    plt.savefig(
        f'{RESULT_IMAGE_PATH}ROC_AUC_{name.replace(" ","_")}.png', bbox_inches='tight')
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    #  Flatten y_train and y_test if they are not already 1D
    y_train = y_train.values.ravel() if isinstance(
        y_train, pd.DataFrame) else y_train.ravel()
    y_test = y_test.values.ravel() if isinstance(
        y_test, pd.DataFrame) else y_test.ravel()

    # Grid search a random forest
    logging.info("Grid searching a random forest...")
    rfc = RandomForestClassifier(random_state=42)
    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1)
    cv_rfc.fit(x_train, y_train)

    # generate predictions from the best estimator
    logging.info("generate predictions from the best estimator")
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    # save best model
    logging.info("save best random forest model")
    joblib.dump(cv_rfc.best_estimator_, f'{MODEL_PATH}rfc_model.pkl')

    # train a logistic regression
    logging.info("Training a logistic regression...")
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(x_train, y_train)

    # generate predictions
    logging.info("generate predictions from the logistic regression")
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # save model
    logging.info("save logistic regression model")
    joblib.dump(lrc, f'{MODEL_PATH}lrc_model.pkl')

    # generate and save classification report
    logging.info("generate and save classification report")

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_test_preds_lr,
                                model_name="Logistic Regression")

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_rf,
                                y_test_preds_rf,
                                model_name="Random Forest")

    feature_importance_plot(cv_rfc, x_test)

    roc_auc_plot(cv_rfc, x_test, y_test, name="Random Forest")
    roc_auc_plot(lrc, x_test, y_test, name="Logistic Regression")


if __name__ == "__main__":
    start_time = time.time()
    df_main = import_data(DATA_FILE_PATH)
    df_main = perform_eda(df_main)
    df_main = encoder_helper(df_main, cat_columns, response="Churn")
    x_tr, x_te, y_tr, y_te = perform_feature_engineering(df_main)
    train_models(x_tr, x_te, y_tr, y_te)
    logging.info("Total time taken: %s seconds", time.time() - start_time)
