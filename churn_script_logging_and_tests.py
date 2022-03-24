'''
Testing & Logging exercise solution

author: Liliana B
date: March 22, 2022
'''
import os
import logging
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        churn_df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert churn_df.shape[0] > 0
        assert churn_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        churn_df = cl.import_data("./data/bank_data.csv")
        logging.info("Testing perform_eda loading data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda: The file wasn't found")
        raise err

    try:
        perform_eda(churn_df)
        images_folder = "images/"
        images_files = ["Churn.jpg", "Correlation_heatmap.jpg",
                        "Customer_Age.jpg", "Marital_Status.jpg",
                        "Total_Trans_Ct.jpg"]

        for file in images_files:
            assert os.path.isfile(images_folder + file)

        logging.info("Testing perform_eda: SUCCESS")

    except AssertionError as err:
        logging.error("Testing perform_eda: image file does not exist")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":
    test_import(cl.import_data)
    test_eda(cl.perform_eda)
