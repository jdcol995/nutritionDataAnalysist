import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.simplefilter("ignore")

def print_start_message():
    """
        This function only prints a message for the 
        start of the code
    """
    print("==============================================")
    print("               RUNNING STARTED                ")
    print("==============================================")
    print("\nThe following steps are going to be executed in order:\n")
    steps = [
        "1. Load Data",
        "2. Clean and Normalize Data",
        "3. Spliting Data"
        "4. Create VAE",
        "5. Train VAE",
        "6. Generate Synthetic Data",
        "7. Extract Top 10 Features using MI, MDI, SF",
        "8. Classify the Data"
    ]
    for step in steps:
        print(step)
    print("\n==============================================")


# Function to read the original data given the path
# and returns the data frame with the original data
def load_data(path):
    data = pd.read_csv(path)
    return data

# Function to clean an normalize the data, it returns a data frame
# with the normalize and clean data
def clean_Normalize(data):
    new_data = data[0:24]
    label = new_data['label']
    features = new_data.drop(['label'],axis=1)
    for column in features.columns:
        features[column].fillna(features[column].median(),inplace=True)
    scaler = MinMaxScaler()
    featuresNormalized = scaler.fit_transform(features)
    featuresNormalized = pd.DataFrame(featuresNormalized,columns=features.columns)
    return featuresNormalized, label

def print_matrix_table():
    """
        This function only prints a matrix that shows  
        how the data is separate in different data sets
        to evaluate the synthetic data
    """
    print('\nStarting classification models with the following data sets:\n')
    # Header
    print("+-----------------+-----------------+")
    print("|    Train Data   |    Test Data    |")
    print("+-----------------+-----------------+")
    # Rows
    data = [
        ("Real", "Real"),
        ("Real", "Synthetic"),
        ("Synthetic", "Real"),
        ("Synthetic", "Synthetic"),
    ]

    for train, test in data:
        print(f"| {train.center(15)} | {test.center(15)} |")
        print("+-----------------+-----------------+")

def evaluate_classifier(classifier, feature_sets, labels, scenarios):
    """
        This function it's to evaluate the classifiers with the different data sets
        to evaluate the classifier with the synthetic data
        it takes four parameters
        classifier -> classifier to evaluate (function)
        feaature_sets -> dictionary with the sets divided as Real and Synthetic
        labels -> dictionary with labels divided as Real and Synthetic
        scenarios -> Dictionary divided by key for the differente scenarios i.e Real-Real, Real-Synthetic
    """
    acc_list = []
    auroc_list = []
    
    for scenario, (train_set, test_set) in scenarios.items():
        train_features = feature_sets[train_set]
        test_features = feature_sets[test_set]
        train_labels = labels[train_set]
        test_labels = labels[test_set]
        acc, auroc = classifier(train_features, train_labels, test_features, test_labels)
        print(f'{scenario}: Accuracy score of {acc:.3f} and AUROC Score of {auroc:.3f}')
        acc_list.append(acc)
        auroc_list.append(auroc)
    return acc_list, auroc_list