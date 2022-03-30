'''
Example of running the Churn library

author: Liliana Badillo
date: March 30, 2022
'''

import joblib
import churn_library as cl
import constants as constants_project


def main():
    '''
      Runs an example of the churn library

      input: None

      output: None
    '''
    # Open the data file
    churn_df = cl.import_data(constants_project.DATA_FILE)

    # Perform EDA and save plots
    cl. perform_eda(churn_df)

    # Encode categorical features
    churn_df = cl.encoder_helper(churn_df, constants_project.CATEGORIC_FEATURES_LST,
                                 constants_project.RESPONSE)

    # Perform feature engineering
    x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
        churn_df, constants_project.RESPONSE)

    # Train the logistic regresssion and random forest models
    cl.train_models(x_train, x_test, y_train, y_test)

    # Load the models
    rfc_model = joblib.load(constants_project.MODELS_FOLDER +
                            constants_project.MODELS_FILES_LST[0])

    lr_model = joblib.load(constants_project.MODELS_FOLDER +
                           constants_project.MODELS_FILES_LST[0])

    cl.feature_importance_plot(rfc_model, x_train,
                               constants_project.IMAGES_FOLDER)

    # Obtain predictions of both models for test data
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

    # Create and store the classification report
    cl.classification_report_image(**predictions)


if __name__ == "__main__":
    main()
