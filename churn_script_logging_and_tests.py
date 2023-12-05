"""Module to test the churn_library.py module

Author: Jeroen van Lier
Date: 2023-12-05
"""

import os
import logging
import pytest

import pandas as pd
import numpy as np
from constants import DATA_FILE_PATH, EDA_IMAGE_PATH, RESULT_IMAGE_PATH, cat_columns


from churn_library import (
    import_data, Plotter, perform_eda, encoder_helper,
    perform_feature_engineering,
    train_models, classification_report_image
)

logging.basicConfig(
    filename='./logs/pytest_logs.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    This function tests the import_data function

    input:
                    None

    output:
                    None
    '''
    pth = pytest.import_path
    assert os.path.isfile(pth)

    try:
        df = import_data(pth)
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    # pytest.import_df = df


def assert_image_exists(plot_method, image_path, assert_exists=True):
    """
    Asserts that the image exists in the folder (optional), then removes all .png files \
        from the folder

    input:
                    plot_method: str, the name of the plot method
                    image_path: str, the path to the image folder
                    assert_exists: bool, whether to assert that the image exists or not

    output:
                    None
    """
    if assert_exists:
        try:
            assert len(os.listdir(image_path)) > 0
            logging.info(
                "Testing plotter: The images appear to have been saved to the folder for plot %s",
                plot_method)
        except AssertionError as err:
            logging.error(
                "AssertionError: Testing plotter: The images don't appear to have been saved \
                    to the folder for plot %s", plot_method)
            raise err

    # remove files from folder
    for fig in os.listdir(image_path):
        logging.info("Testing plotter: Removing file %s", fig)
        os.remove(f'{image_path}{fig}')


def test_plotter():
    '''
    This function tests the plotter class

    input:
                    None

    output:
                    None
    '''

    # check if folder exists to save figures
    assert os.path.isdir(EDA_IMAGE_PATH)

    # generate dummy data of a normal distribution for 2 columns
    col1_data = np.random.normal(size=1000)
    col2_data = np.random.normal(size=1000)

    # Create the DataFrame
    dummy_df = pd.DataFrame({'col1': col1_data, 'col2': col2_data})

    plotter = Plotter()
    # test barplot
    plotter.bar_plot(dummy_df['col1'])
    assert_image_exists("bar_plot", EDA_IMAGE_PATH)
    # test histplot
    plotter.hist_plot(dummy_df['col1'])
    assert_image_exists("hist_plot", EDA_IMAGE_PATH)
    # test snshistplot
    plotter.snshist_plot(dummy_df['col1'])
    assert_image_exists("snshist_plot", EDA_IMAGE_PATH)
    # test heatmap
    plotter.heatmap_plot(dummy_df)
    assert_image_exists("heatmap_plot", EDA_IMAGE_PATH)


def test_eda():
    '''
    Tests the perform_eda function

    input:
                    None

    output:
                    None
    '''
    # check if folder exists to save figures
    assert os.path.isdir(EDA_IMAGE_PATH)
    assert_image_exists("all_plots", EDA_IMAGE_PATH, assert_exists=False)

    eda_df = perform_eda(pytest.import_df)

    try:
        assert eda_df is not None
        assert eda_df.shape[0] > 0
        assert eda_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "AssertionError: Testing perform_eda: The dataframe doesn't \
                appear to have rows and columns")
        raise err

    try:
        assert_image_exists("all_plots", EDA_IMAGE_PATH)
    except AssertionError as err:
        logging.error(
            "AssertionError: Testing perform_eda: The images don't appear \
                to have been saved to the folder")
        raise err

    try:

        assert "churn" in [x.lower() for x in eda_df.columns]
    except AssertionError as err:
        logging.error(
            "AssertionError: Testing perform_eda: The target column churn \
                is not present in the dataframe")
        raise err

    pytest.eda_df = eda_df


def test_encoder_helper():
    '''
    Tests the encoder_helper function

    input:
                    None

    output:
                    None
    '''
    try:
        assert len(cat_columns) > 0
    except AssertionError as err:
        logging.error(
            "AssertionError: Testing encoder_helper: The cat_columns list is empty")
        raise err
    try:
        assert isinstance(cat_columns, list)
    except AssertionError as err:
        logging.error(
            "AssertionError: Testing encoder_helper: The cat_columns is not a list")
        raise err

    total_cols = len(pytest.eda_df.columns) + len(cat_columns)
    encode_df = encoder_helper(
        pytest.eda_df,
        cat_columns,
        response='Churn')

    try:
        assert encode_df.shape[1] == total_cols
    except AssertionError as err:
        logging.error(
            "AssertionError: Testing encoder_helper: The dataframe doesn't appear to \
                have the correct number of columns")
        raise err

    pytest.encode_df = encode_df


def test_perform_feature_engineering():
    '''
    Tests the perform_feature_engineering function

    input:
                    None

    output:
                    None
    '''
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        pytest.encode_df, response="Churn")

    try:
        assert x_train is not None
        assert x_test is not None
        assert y_train is not None
        assert y_test is not None
    except AssertionError as err:
        logging.error(
            "AssertionError: Testing perform_feature_engineering: The X and y dataframes \
                are not present")
        raise err

    try:
        assert x_train.shape[0] > 0
        assert x_train.shape[1] > 0
        assert x_test.shape[0] > 0
        assert x_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
    except AssertionError as err:
        logging.error(
            "AssertionError: Testing perform_feature_engineering: The X and y dataframes \
                don't appear to have rows and columns")
        raise err


def test_classification_report_image():
    '''
    Tests the classification_report_image function

    input:
                    None

    output:
                    None
    '''
    # check if folder exists to save figures
    assert os.path.isdir(RESULT_IMAGE_PATH)
    assert_image_exists(
        "classification_report_image",
        RESULT_IMAGE_PATH,
        assert_exists=False)

    classification_report_image(
        pytest.y_train,
        pytest.y_test,
        pytest.y_train.sample(
            frac=1).reset_index(
            drop=True),
        pytest.y_test.sample(
            frac=1).reset_index(
            drop=True),
        "test")
    assert_image_exists("classification_report_image", RESULT_IMAGE_PATH)


def test_train_models():
    '''
    Tests the train_models function

    input:
                    None

    output:
                    None
    '''
    train_models(pytest.x_train, pytest.x_test, pytest.y_train, pytest.y_test)

    assert_image_exists("train_models", RESULT_IMAGE_PATH, assert_exists=False)


if __name__ == "__main__":

    test_import()
    test_plotter()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
    test_classification_report_image()
