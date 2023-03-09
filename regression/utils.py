import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split

def loadDataset(
    features = [
        'Penicillin V Potassium 500 MG',
        'Computed tomography of chest and abdomen',
        'Plain chest X-ray (procedure)', 
        'Low Density Lipoprotein Cholesterol',
        'Creatinine',
        'AGE_DIAGNOSIS'
    ],
    split_percent=None,
    split_seed=42
):
    """

    YOU DO NOT NEED TO MODIFY THIS FUNCTION.
    Load NSCLC dataset as NumPy arrays.

    Arguments:
        features (list): Features to pull from the data. (The full list of potential features is below.)
        split_percent (float, None): Percentage of data to use for training. (Optional.)
        split_seed (int): Seed to use for randomly splitting the data. (Optional.)
    
    Output:
        X (np.ndarray) and y (np.array), corresponding to dataset and labels, respectively.
        If split_percent is specified, returns X_train, X_test, y_train, y_test.

    --------

    List of potential features. NSCLC is the classification label, where 1 = NSCLC and 0 = small cell lung cancer.

    'NSCLC', 'GENDER', 'Penicillin V Potassium 250 MG', 'Penicillin V Potassium 500 MG',
    'Computed tomography of chest and abdomen', 'Plain chest X-ray (procedure)', 'Diastolic Blood Pressure',
    'Body Mass Index', 'Body Weight', 'Body Height', 'Systolic Blood Pressure',
    'Low Density Lipoprotein Cholesterol', 'High Density Lipoprotein Cholesterol', 'Triglycerides',
    'Total Cholesterol', 'Documentation of current medications',
    'Fluticasone propionate 0.25 MG/ACTUAT / salmeterol 0.05 MG/ACTUAT [Advair]',
    '24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet',
    'Carbon Dioxide', 'Hemoglobin A1c/Hemoglobin.total in Blood', 'Glucose', 'Potassium', 'Sodium', 'Calcium',
    'Urea Nitrogen', 'Creatinine', 'Chloride', 'AGE_DIAGNOSIS'

    """

    # Read dataset
    data_dir = pathlib.Path(__file__).resolve().parent.parent / 'data'
    full_df = pd.read_csv(data_dir / "nsclc.csv", index_col="ID")
    
    # Always include the class label
    if "NSCLC" not in features: 
        features.append("NSCLC")
    
    # Select desired features
    full_df = full_df.loc[:,features]
    
    # Split and return appropriate datasets
    X = full_df.loc[:, full_df.columns != 'NSCLC'].values
    y = full_df["NSCLC"].values
    
    if split_percent is not None: 
        return train_test_split(X, y, train_size=split_percent, random_state=split_seed)

    return X,y