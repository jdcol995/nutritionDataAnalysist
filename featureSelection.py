from sklearn.feature_selection import SelectKBest, mutual_info_classif
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import pandas as pd
import time
import numpy as np

class FEATURE_SELECTION:
    def __init__(self, X_train, y_train, X_test, y_test, X_train_syn, y_train_syn, X_test_syn, y_test_syn):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_train_syn = X_train_syn
        self.y_train_syn = y_train_syn
        self.X_test_syn = X_test_syn
        self.y_test_syn = y_test_syn

    def mutual_Info_selector(self,K):
        """
            Function to select the top K more representative features
            using mutual information method, it only takes one parameter
            K
        """
        selector = SelectKBest(score_func=mutual_info_classif, k=K)
        start = time.time()
        print(f"Estimating the top {K} features with Mutual Information feature selector...")
        selector.fit_transform(self.X_train, self.y_train)
        selected_features_filter = np.where(selector.get_support())[0]
        plt.figure(figsize=(12, 6))
        plt.bar(range(K), selector.scores_[selected_features_filter], tick_label=self.X_train.columns[selected_features_filter])
        plt.xticks(rotation=45, fontsize=8)
        plt.title('Top 10 Features using Filter Method Mutual Information')
        plt.xlabel('Feature Index')
        plt.ylabel('Mutual Information Score')
        miFeatures = []
        for i in selected_features_filter:
            miFeatures.append(self.X_train.columns[i])
        featuresMutualInfoTrain = self.X_train[miFeatures]
        featuresMutualInfoSyncTrain = self.X_train_syn[miFeatures]
        featuresMutualInfoTest = self.X_test[miFeatures]
        featuresMutualInfoSynTest = self.X_test_syn[miFeatures] 
        end = time.time()
        print(f'...........Done in {end -start:.2f} seconds')       
        return featuresMutualInfoTrain, featuresMutualInfoSyncTrain, featuresMutualInfoTest, featuresMutualInfoSynTest

    def mdi_selector(self, K):
        """
            Function to select the top K more representative features
            using MDI method, it only takes one parameter
            K
        """
        clf = RandomForestClassifier(n_estimators=500, random_state=42)
        start = time.time()
        print(f"Estimating the top {K} features with MDI feature selector...")
        clf.fit(self.X_train,self.y_train)
        importances = clf.feature_importances_
        featuresSortedDataFrame = pd.DataFrame({'Feature':self.X_train.columns,'Importance':importances})
        featuresSortedDataFrame.sort_values('Importance',ascending=False,inplace=True)
        featureImportance = featuresSortedDataFrame['Feature']
        # visualizar datos
        top50Features = featuresSortedDataFrame.head(K)
        plt.figure(figsize=(9,6))
        plt.barh(top50Features['Feature'], top50Features['Importance'], align='center', color='blue')
        plt.xlabel('Importancia')
        plt.ylabel('Característica')
        plt.title('Importancia de las 10 características más importantes')
        plt.gca() # Ordenar las características de mayor a menor importancia
        featuresMDI_Train = self.X_train[featureImportance[0:K]].reset_index(drop=True)
        featuresMDI_synTrain = self.X_train_syn[featureImportance[0:K]].reset_index(drop=True)
        featuresMDI_Test = self.X_test[featureImportance[0:K]].reset_index(drop=True)
        featuresMDI_synTest = self.X_test_syn[featureImportance[0:K]].reset_index(drop=True)
        end = time.time()
        print(f'...........Done in {end -start:.2f} seconds')         
        return featuresMDI_Train, featuresMDI_synTrain, featuresMDI_Test, featuresMDI_synTest

    def sequential_selector(self,K):
        """
        Function to extract the top K features using Sequential method with forward selection
        It take K as input and returns four dataframes
        """
        SFS = SequentialFeatureSelector(LogisticRegression(),
                                k_features = K,
                                forward = True,
                                scoring = 'r2')
        print(f"Estimating the top {K} features with Sequential Forward feature selector...")
        start = time.time()
        SFS.fit(self.X_train,self.y_train)
        SFS_results = pd.DataFrame(SFS.subsets_).transpose()
        print(SFS_results)
        forwardFeatures = []
        forwardFeatures_syn = []
        for i in SFS_results['feature_idx'][K]:
            forwardFeatures.append(self.X_train.columns[i])
            forwardFeatures_syn.append(self.X_train_syn.columns[i])
            print(f'feature: {self.X_train.columns[i]}\nfeature synthethic: {self.X_train_syn.columns[i]}')
        featuresForward_Train = self.X_train[forwardFeatures[:]]
        featuresForward_synTrain = self.X_train_syn[forwardFeatures_syn[:]]
        featuresForward_Test = self.X_test[forwardFeatures_syn[:]]
        featuresForward_synTest = self.X_test_syn[forwardFeatures_syn[:]]
        end = time.time()
        print(f'...........Done in {end -start:.2f} seconds')    
        return featuresForward_Train, featuresForward_synTrain, featuresForward_Test, featuresForward_synTest