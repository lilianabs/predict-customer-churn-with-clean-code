'''
Constants for the churn project

author: Liliana Badillo
date: March 30, 2022
'''

DATA_FILE = "./data/bank_data.csv"

RESPONSE = "Churn"
CATEGORY_LST = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category',
                    'Card_Category']

IMAGES_FOLDER = "images/"
IMAGES_FILES_LST = ["Churn.jpg", "Correlation_heatmap.jpg",
                    "Customer_Age.jpg", "Marital_Status.jpg",
                    "Total_Trans_Ct.jpg"]

MODELS_FOLDER = "models/"
MODELS_FILES_LST = ["logistic_model.pkl", "rfc_model.pkl"]

NUMERIC_FEATURES_LST = ["Churn", "Customer_Age", "Total_Trans_Ct"]
CATEGORIC_FEATURES_LST = ["Marital_Status"]

CATEGORIES_FEATURE_ENG_LST = ['Gender', 'Education_Level', 'Marital_Status', 
                              'Income_Category', 'Card_Category']

COLS_TRAINING_LST = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                     'Total_Relationship_Count', 'Months_Inactive_12_mon',
                     'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                     'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                     'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

MODELS_LST = ["lr", "rf"]