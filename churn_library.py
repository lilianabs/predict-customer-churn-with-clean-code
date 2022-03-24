'''
Churn library

author: Liliana Badillo
date: March 22, 2022
'''
import logging
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


logging.basicConfig(
    filename='logs/churn_library.log',
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
        churn_df = pd.read_csv(pth)
        logging.info("import_data: Loading CSV file SUCCESS")
        return churn_df
    except FileNotFoundError:
        logging.error("import_data: CSV file not found: %s", pth)


def plot_numeric_feature(feature):
    '''
    plots a histplot (numeric feature)
    and saves figure to images folder

    input:
          feature: pandas series
    output:
          None
    '''
    plt.figure(figsize=(20, 10))
    sns.histplot(feature)
    plt.title(str(feature.name))
    plt.savefig('images/{}.jpg'.format(feature.name))


def plot_categoric_feature(feature):
    '''
    plots a bar plot (categoric feature)
    and saves figure to images folder

    input:
          feature: pandas series
    output:
          None
    '''
    plt.figure(figsize=(20, 10))
    feature.value_counts('normalize').plot(kind='bar')
    plt.title(str(feature.name))
    plt.savefig('images/{}.jpg'.format(feature.name))


def plot_correlation(churn_df):
    '''
    plots a correlation heatmap of numeric features
    and saves figure to images folder

    input:
          churn_df: pandas dataframe
    output:
          None
    '''
    plt.figure(figsize=(20, 10))
    sns.heatmap(churn_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Correlation Heatmap')
    plt.savefig('images/Correlation_heatmap.jpg')


def perform_eda(churn_df):
    '''
    performs eda on df and saves figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    try:
        # Add Churn column to the dataframe
        churn_df['Churn'] = churn_df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        logging.info("perform_eda: Added Churn column to dataframe")
    except Exception as err:
        logging.error("perform_eda: Could not add Churn column to dataframe: %s", err)

    numeric_features_lst = ["Churn", "Customer_Age", "Total_Trans_Ct"]
    categoric_features_lst = ["Marital_Status"]

    # Plot numeric columns and save images
    for feature in numeric_features_lst:
        try:
            plot_numeric_feature(churn_df[feature])
            logging.info("Numeric plot stored: %s", feature)
        except Exception as err:
            logging.error("Clould not store numeric plot: %s", feature)

    # Plot categoric columns and save images
    for feature in categoric_features_lst:
        try:
            plot_categoric_feature(churn_df[feature])
            logging.info("Categoric plot stored: %s", feature)
        except Exception as err:
            logging.error("Could not store categoric plot: %s", feature)

    # Plot correlation heatmap
    try:
        plot_correlation(churn_df)
        logging.info("Correlation plot stored")
    except Exception as err:
        logging.error("Could not store correlation plot: %s", err)


def encoder_helper(churn_df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            churn_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                      for naming variables or index y column]

    output:
            churn_df: pandas dataframe with new columns
    '''
    try:
        # Add Churn column to the dataframe
        churn_df[response] = churn_df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        logging.info("encoder_helper: Added Churn column to dataframe")
    except Exception as err:
        logging.error(
            "encoder_helper: Could not add Churn column to dataframe: %s", err)

    for category in category_lst:
        try:
            category_groups = churn_df.groupby(category).mean()[response]
            category_new_feature = [category_groups.loc[val]
                                    for val in churn_df[category]]
            churn_df[category + "_" + response] = category_new_feature
            logging.info(
                "encoder_helper: Added new categorical column %s",
                category +
                response)
        except Exception as err:
            logging.error(
                "encoder_helper: Could not add categorical column: %s",
                category +
                "_" +
                response)

    return churn_df


def perform_feature_engineering(churn_df, response):
    '''
    input:
          churn_df: pandas dataframe
          response: string of response name [optional argument that
                    could be used for naming variables or index y column]

    output:
           x_train: X training data
           x_test: X testing data
           y_train: y training data
           y_test: y testing data
    '''
    try:
        # Add Churn column to the dataframe
        churn_df[response] = churn_df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        logging.info(
            "perform_feature_engineering: Added Churn column to dataframe")
    except Exception as err:
        logging.error(
            "perform_feature_engineering: Could not add Churn column to dataframe %s ", err)

    predictor = churn_df[response]

    category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category',
                    'Card_Category']

    churn_df = encoder_helper(churn_df, category_lst, response)

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

    new_categoric_cols = [category + "_" +
                          response for category in category_lst]

    features = churn_df[keep_cols + new_categoric_cols]

    x_train, x_test, y_train, y_test = train_test_split(
        features, predictor, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
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

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass

# if __name__ == "__main__":
#    df = import_data("data/bank_data.csv")
#    perform_eda(df)
#    category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category',
#                     'Card_Category'
#    ]
#    response = "Churn"
#    encoder_helper(df, category_lst, response)
#    print(df.info())
#    print(df.head())
