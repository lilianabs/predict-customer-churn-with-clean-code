'''
Testing & Logging exercise solution

author: Liliana Badillo
date: March 22, 2022
'''
import os
import logging
import joblib

import constants
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
        churn_df = import_data(constants.DATA_FILE)
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
        churn_df = cl.import_data(constants.DATA_FILE)
        logging.info("Testing perform_eda loading data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda: The file wasn't found")
        raise err

    try:
        perform_eda(churn_df)

        for file in constants.IMAGES_FILES_LST:
            logging.info("Testing perform_eda file %s", file)
            assert os.path.isfile(constants.IMAGES_FOLDER + file)

        logging.info("Testing perform_eda: SUCCESS")

    except AssertionError as err:
        logging.error("Testing perform_eda: image file does not exist %s", err)
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        churn_df = cl.import_data(constants.DATA_FILE)
        logging.info("Testing encoder_helper: loading data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing encoder_helper: The file wasn't found")
        raise err

    try:
        # Get number of original columns
        num_cols_churn_df = churn_df.shape[1]
        churn_df = encoder_helper(churn_df, constants.CATEGORY_LST,
                                  constants.RESPONSE)

        # Check that we have all of the new categoric columns and the response
        assert churn_df.shape[1] == (
            num_cols_churn_df + len(constants.CATEGORY_LST) + 1)

        logging.info(
            "Testing encoder_helper: correct number of columns: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: error not enough categorical features")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        churn_df = cl.import_data(constants.DATA_FILE)
        logging.info(
            "Testing perform_feature_engineering: loading data: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing perform_feature_engineering: The file wasn't found")
        raise err

    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            churn_df, constants.RESPONSE)

        assert x_train.shape[0] > 0
        assert x_train.shape[1] > 0

        assert x_test.shape[0] > 0
        assert x_test.shape[1] > 0

        assert y_train.shape[0] == x_train.shape[0]
        assert y_test.shape[0] == x_test.shape[0]

        logging.info(
            "Testing perform_feature_engineering: created features for training: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: error creating features for training")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        churn_df = cl.import_data(constants.DATA_FILE)
        logging.info(
            "Testing train_models: loading data: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing train_models: The file wasn't found")
        raise err

    try:
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
            churn_df, constants.RESPONSE)

        train_models(x_train, x_test, y_train, y_test)

        # Check the ROC curve plot was stored
        assert os.path.isfile(constants.IMAGES_FOLDER + "roc_curve.jpg")

        # Check the models were stored
        for model in constants.MODELS_FILES_LST:
            assert os.path.isfile(constants.MODELS_FOLDER + model)

        logging.info(
            "Testing train_models: created models and stored roc curve plot: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: error training models and storing roc curve plot ")
        raise err


def test_feature_importance_plot(feature_importance_plot):
    '''
    test feature_importance_plot
    '''
    try:
        churn_df = cl.import_data(constants.DATA_FILE)
        logging.info(
            "Testing feature_importance_plot: loading data: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing feature_importance_plot: The file wasn't found")
        raise err

    try:
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
            churn_df, constants.RESPONSE)

        # Load the model
        rfc_model = joblib.load(constants.MODELS_FOLDER + 'rfc_model.pkl')

        # Create the feature importance model
        feature_importance_plot(rfc_model, x_train, constants.IMAGES_FOLDER)
        logging.info(
            "Testing feature_importance_plot: created feature importance plot: SUCCESS")

        # Check the feature importance plot was stored
        assert os.path.isfile(
            constants.IMAGES_FOLDER +
            "feature_importance.jpg")

        logging.info(
            "Testing feature_importance_plot: stored feature importance plot: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot: error creating feature importance plot %s", err)
        raise err


def test_classification_report_image(classification_report_image):
    '''
    test classification_report_image
    '''
    try:
        churn_df = cl.import_data(constants.DATA_FILE)
        logging.info(
            "Testing classification_report_image: loading data: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing classification_report_image: The file wasn't found")
        raise err

    try:
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
            churn_df, constants.RESPONSE)

        # Load the model
        rfc_model = joblib.load(constants.MODELS_FOLDER + 'rfc_model.pkl')
        lr_model = joblib.load(constants.MODELS_FOLDER + 'logistic_model.pkl')

        y_train_preds_rf = rfc_model.predict(x_train)
        y_test_preds_rf = rfc_model.predict(x_test)

        y_train_preds_lr = lr_model.predict(x_train)
        y_test_preds_lr = lr_model.predict(x_test)

        predictions = {"y_train_preds_rf": y_train_preds_rf,
                       "y_test_preds_rf": y_test_preds_rf,
                       "y_train_preds_lr": y_train_preds_lr,
                       "y_test_preds_lr": y_test_preds_lr,
                       "y_train": y_train,
                       "y_test": y_test}

        classification_report_image(**predictions)

        # Check the classification report images were stored
        assert os.path.isfile(
            constants.IMAGES_FOLDER +
            "classification_report_lr.jpg")
        assert os.path.isfile(
            constants.IMAGES_FOLDER +
            "classification_report_rf.jpg")

        logging.info(
            "Testing classification_report_image: stored classification report images: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing classification_report_image: error classification report image %s", err)
        raise err


if __name__ == "__main__":
    test_import(cl.import_data)
    test_eda(cl.perform_eda)
    test_encoder_helper(cl.encoder_helper)
    test_perform_feature_engineering(cl.perform_feature_engineering)
    test_train_models(cl.train_models)
    test_feature_importance_plot(cl.feature_importance_plot)
    test_classification_report_image(cl.classification_report_image)
