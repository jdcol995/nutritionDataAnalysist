from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def support_vector_machine(X_train,y_train, X_test,y_test):
    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    y_pred_proba_svm = svm_model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_pred_proba_svm)
    accuracy = accuracy_score(y_test, y_pred_svm)
    return accuracy, auroc

def random_forest(X_train,y_train, X_test,y_test):
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_pred_proba_rf)
    accuracy = accuracy_score(y_test, y_pred_rf)
    return accuracy, auroc

def multi_layer_perceptron(X_train,y_train, X_test,y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    y_pred_proba_mlp = mlp.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_pred_proba_mlp)
    accuracy = accuracy_score(y_test, y_pred_mlp)
    return accuracy, auroc