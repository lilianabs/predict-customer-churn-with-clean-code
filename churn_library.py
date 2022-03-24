'''
Churn library

author: Liliana Badillo
date: March 22, 2022
'''

import logging
logging.basicConfig(
    filename='logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    try:
        df = pd.read_csv(pth)
        logging.info("import_data: Loading CSV file SUCCESS")
        return df
    except FileNotFoundError:
        logging.error("import_data: CSV file not found: " + pth)


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
    sns.histplot(feature);
    plt.title(str(feature.name));
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
    feature.value_counts('normalize').plot(kind='bar');
    plt.title(str(feature.name));
    plt.savefig('images/{}.jpg'.format(feature.name)) 


def plot_correlation(df):
    '''
    plots a correlation heatmap of numeric features
    and saves figure to images folder

    input: 
          df: pandas dataframe
    output:
          None
    '''
    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.title('Correlation Heatmap');
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
        churn_df['Churn'] = churn_df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        logging.info("perform_eda: Added Churn column to dataframe")
    except:
        logging.error("perform_eda: Could not add Churn column to dataframe")

    numeric_features_lst = ["Churn", "Customer_Age", "Total_Trans_Ct"]
    categoric_features_lst = ["Marital_Status"]

    # Plot numeric columns and save images
    for feature in numeric_features_lst:
        try:
            plot_numeric_feature(churn_df[feature])
            logging.info("Numeric plot stored: " + feature)
        except:
            logging.error("Clould not store numeric plot: " + feature)
    
    # Plot categoric columns and save images
    for feature in categoric_features_lst:
        try:
            plot_categoric_feature(churn_df[feature])
            logging.info("Categoric plot stored: " + feature)
        except:
            logging.error("Could not store categoric plot: " + feature)

    # Plot correlation heatmap
    try:
        plot_correlation(churn_df)
        logging.info("Correlation plot stored")
    except:
        logging.error("Could not store correlation plot")



def encoder_helper(churn_df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            churn_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            churn_df: pandas dataframe with new columns
    '''
    try:
        # Add Churn column to the dataframe
        churn_df['Churn'] = churn_df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        logging.info("encoder_helper: Added Churn column to dataframe")
    except:
        logging.error("encoder_helper: Could not add Churn column to dataframe")
         
    for category in category_lst:
        try:
            category_groups = churn_df.groupby(category).mean()['Churn']
            category_new_feature = [category_groups.loc[val] for val in churn_df[category]]
            churn_df[category + response] = category_new_feature
            logging.info("encoder_helper: Added new categorical column " + category + response)
        except:
            logging.error("encoder_helper: Could not add categorical column: " + category + response)
    
    return churn_df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

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

#if __name__ == "__main__":
#    df = import_data("data/bank_data.csv")
#    perform_eda(df)
#    category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category',
#                     'Card_Category'
#    ]
#    response = "_Churn"
#    encoder_helper(df, category_lst, response)
#    print(df.info())
