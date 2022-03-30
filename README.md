# Churn library

The Churn library consists of two machine learning models for predicting customer churn. Currenly, it contains logistic regression and random forest.

## Running the project

Before you run this project, make sure you satisfy the following requirements:

* Python >= 3.8
* Conda

To run this project locally, follow the next steps:

1. Create a conda environment:

    ```
    conda create --name {env_name} python==3.8
    ```

    Note: Set a name for the env_name variable.

2. Activate the conda environment:

    ```
    conda activate env_name
    ```

3. Install the necesary libraries:

    ```
    conda install --file requirements.txt
    ```

4. Run an example of the Churn library:

   ```
    python client_churn_library.py
   ```


## Contents of the project

This project contains the following folders:

* **data:** It stores the data used for training the models.
* **images:** It stores the plots that Churn library creates during EDA.
* **logs:** It stores the Churn library logs.
* **models:** It stores the `.pkl` files of the models.

## Running the tests

To run the tests of the Churn library:

```
python churn_script_logging_and_tests.py
```
