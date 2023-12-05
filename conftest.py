"""Pytest configuration file for the project.

Author: Jeroen van Lier

Date: 2023-12-05
"""

import pytest
import numpy as np
import pandas as pd
from constants import DATA_FILE_PATH, keep_cols, churn_column, cat_columns, quant_columns
assert len(churn_column) == 1
churn_column = churn_column[0]


def create_test_dataframe(df_type=None):
    """
    Create and return a sample pandas DataFrame for testing purposes.

    input:
        columns: number of columns in the DataFrame

    output:
        pd.DataFrame: A sample DataFrame with test data.
    """
    
    if df_type == 'x':
        return pd.DataFrame(np.random.randn(100, len(keep_cols)), columns=keep_cols)
        
    elif df_type == 'y':
        return pd.DataFrame(np.random.randint(2, size=(100, 1)), columns=['Churn'])
        
    elif df_type == 'eda':
        dummy_df_cat = pd.DataFrame(np.random.randn(100, len(cat_columns)), columns=cat_columns)
        # create a df with catagorical columns of either "A" or "B" for each column
        dummy_df_cat = dummy_df_cat.applymap(lambda x: np.random.choice(["A", "B"]))
        # create a df with continous columns of random numbers
        dummy_df_quant = pd.DataFrame(np.random.randn(100, len(quant_columns)), columns=quant_columns) 
        # create a df with a column of string type continaing either "Existing Customer" or "Attrited Customer"
        status_values = ["Existing Customer", "Attrited Customer"]
        # Randomly select values from the list to create the column data
        customer_status = np.random.choice(status_values, size=100)
        # Create the DataFrame
        df_churn_columns = pd.DataFrame({churn_column: customer_status})

        return pd.concat([dummy_df_cat, dummy_df_quant, df_churn_columns], axis=1)
    
    else:
        raise ValueError("df_type must be either 'x', 'y', or 'eda'")




def df_plugin():

    return None

# Creating a Dataframe object in Namespace


def pytest_configure():
    """
    Create a dataframe object in the pytest namespace.
    """
    pytest.import_df = create_test_dataframe(df_type='eda')
    pytest.eda_df = create_test_dataframe(df_type='eda')
    pytest.encode_df = create_test_dataframe(df_type='eda')
    pytest.x_train = create_test_dataframe(df_type='x')
    pytest.x_test = create_test_dataframe(df_type='x')
    pytest.y_train = create_test_dataframe(df_type='y')
    pytest.y_test = create_test_dataframe(df_type='y')
    pytest.import_path = DATA_FILE_PATH
