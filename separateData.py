from sklearn.model_selection import train_test_split

def general_spliting(data,label,testSize = 0.2):
    """
    function to split the data, it only renames the train_test_split from
    sklearn
    """
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=testSize, random_state=42)
    return X_train, X_test, y_train, y_test