from matplotlib import pyplot as plt
from dataUtils import load_data, clean_Normalize, print_matrix_table, print_start_message, evaluate_classifier
from separateData import general_spliting
from train_generate import train_Vae, generate_data
from featureSelection import FEATURE_SELECTION
from plotDistributions import plot_distributions
from classifiers import support_vector_machine, random_forest, multi_layer_perceptron
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str)
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    print_start_message()
    print("==============================================")
    print("         Loading and Normalizing Data         ")
    path = 'data/' + args.filename  + '.csv' # modify according to your data
    data, label = clean_Normalize(load_data(path))
    print('\n..........Done\n'.center(30))
    # Split for general training and testing data
    print("==============================================")
    print("         Spliting Data         ")
    X_train, X_test, y_train, y_test = general_spliting(data,label, 0.2)
    # Split for VAE training and validation
    X_train_vae, X_test_vae, y_train_vae, y_test_vae = general_spliting(X_train, y_train, 0.2)
    print('\n..........Done\n'.center(30))
    # VAE training
    print("==============================================")
    print("           Creating and Training VAE            ")
    vae_model = train_Vae(X_train_vae,y_train_vae,X_test_vae,y_test_vae,1000)
    # save the model
    print('\n..........Done\n'.center(30))
    print("==============================================")
    print("           Generating synthetic Data            ")
    syntetic_data = generate_data(24,vae_model)
    syntetic_data = pd.DataFrame(syntetic_data,columns = data.columns)
    syntetic_data.to_csv('syntheticData/' + args.filename + '_syncData.csv') # modify to your needs
    # Split the synthetic data
    X_train_syn, X_test_syn, y_train_syn, y_test_syn = general_spliting(syntetic_data, label, 0.2)
    print('\n..........Done\n'.center(30))

    # # Feature Selection methods
    path = 'results/' + args.filename # modify according to your needs
    print("==============================================")
    print("           Extracting Features           \n")    
    K = 10
    feature_selector = FEATURE_SELECTION(X_train=X_train, y_train=y_train,
                                          X_test=X_test, y_test=y_test,
                                          X_train_syn=X_train_syn, y_train_syn=y_train_syn,
                                          X_test_syn=X_test_syn, y_test_syn=y_test_syn)
    # Mutual information method
    featuresMutualInfoTrain, featuresMutualInfoSyncTrain, featuresMutualInfoTest, featuresMutualInfoSynTest = feature_selector.mutual_Info_selector(K)
    plot_distributions(featuresMutualInfoTrain,featuresMutualInfoSyncTrain, 'MI', path)
    # MDI method
    featuresMDI_Train, featuresMDI_synTrain, featuresMDI_Test, featuresMDI_synTest = feature_selector.mdi_selector(K)
    plot_distributions(featuresMDI_Train,featuresMDI_synTrain, 'MDI', path)
    # Sequential Forward features selection
    featuresForward_Train, featuresForward_synTrain, featuresForward_Test, featuresForward_synTest = feature_selector.sequential_selector(K)
    plot_distributions(featuresForward_Train,featuresForward_synTrain, 'SF', path)
    print('\n..........Done\n'.center(30))
    # Save the features for each feature selector method
    features_names_df = pd.DataFrame({
    'Mutual Information': featuresMutualInfoTrain.columns.tolist(),
    'MDI': featuresMDI_Train.columns.tolist(),
    'Sequential Forward': featuresForward_Train.columns.tolist()
    })
    output_file_features = 'results/features_names_' + args.mode + '.csv'
    features_names_df.to_csv(output_file_features, index=False)

    # Classification models
    print("==============================================")
    print("        Running Classification models          \n")    
    print_matrix_table()
    # -------------------------------------------------------------------------------------------
    #                                    MI features
    feature_sets = {
    "real_train": featuresMutualInfoTrain,
    "real_test": featuresMutualInfoTest,
    "syn_train": featuresMutualInfoSyncTrain,
    "syn_test": featuresMutualInfoSynTest,
    "Combined_train": pd.concat([featuresMutualInfoTrain,featuresMutualInfoSyncTrain], ignore_index=True),
    "Combined_test": pd.concat([featuresMutualInfoTest,featuresMutualInfoSynTest], ignore_index=True),
    "Combined_train_original": pd.concat([featuresMutualInfoTrain[:17],featuresMutualInfoSyncTrain[:17]], ignore_index=True),
    "Combined_test_original": pd.concat([featuresMutualInfoTest,featuresMutualInfoTrain[-2:]], ignore_index=True)
    }

    labels = {
        "real_train": y_train,
        "real_test": y_test,
        "syn_train": y_train_syn,
        "syn_test": y_test_syn,
        "Combined_train": pd.concat([y_train,y_train_syn], ignore_index=True),
        "Combined_test": pd.concat([y_test,y_test_syn], ignore_index=True),
        "Combined_train_original": pd.concat([y_train[:17],y_train_syn[:17]], ignore_index=True),
        "Combined_test_original": pd.concat([y_test,y_train[-2:]], ignore_index=True)        
    }

    scenarios = {
        "Real-Real": ("real_train", "real_test"),
        "Real-Synthetic": ("real_train", "syn_test"),
        "Synthetic-Real": ("syn_train", "real_test"),
        "Synthetic-Synthetic": ("syn_train", "syn_test"),
        "Combined": ("Combined_train","Combined_test"),
        "Combined_original": ("Combined_train_original","Combined_test_original")
    }
    # For SVM
    print('\nInitializing SVM classification with MI features\n'.center(15))
    svm_mi_acc, svm_mi_auroc = evaluate_classifier(support_vector_machine, feature_sets, labels, scenarios)

    # For RF
    print('\nInitializing RF classification with MI features\n'.center(15))
    rf_mi_acc, rf_mi_auroc = evaluate_classifier(random_forest, feature_sets, labels, scenarios)

    # For MLP
    print('\nInitializing MLP classification with MI features\n'.center(15))
    mlp_mi_acc, mlp_mi_auroc = evaluate_classifier(multi_layer_perceptron, feature_sets, labels, scenarios)
    #---------------------------------------------------------------------------------------------------
    # #                                    MDI features
    feature_sets = {
    "real_train": featuresMDI_Train,
    "real_test": featuresMDI_Test,
    "syn_train": featuresMDI_synTrain,
    "syn_test": featuresMDI_synTest,
    "Combined_train": pd.concat([featuresMDI_Train,featuresMDI_synTrain], ignore_index=True),
    "Combined_test": pd.concat([featuresMDI_Test,featuresMDI_synTest], ignore_index=True),
    "Combined_train_original": pd.concat([featuresMDI_Train[:17],featuresMDI_synTrain[:17]], ignore_index=True),
    "Combined_test_original": pd.concat([featuresMDI_Test,featuresMDI_Train[-2:]], ignore_index=True)
    }

    # For SVM
    print('\nInitializing SVM classification with MDI features\n'.center(15))
    svm_mdi_acc, svm_mdi_auroc = evaluate_classifier(support_vector_machine, feature_sets, labels, scenarios)

    # For RF
    print('\nInitializing RF classification with MDI features\n'.center(15))
    rf_mdi_acc, rf_mdi_auroc = evaluate_classifier(random_forest, feature_sets, labels, scenarios)

    # For MLP
    print('\nInitializing MLP classification with MDI features\n'.center(15))
    mlp_mdi_acc, mlp_mdi_auroc = evaluate_classifier(multi_layer_perceptron, feature_sets, labels, scenarios)
    #--------------------------------------------------------------------------------------------------
    #                                    Sequential features
    feature_sets = {
    "real_train": featuresForward_Train,
    "real_test": featuresForward_Test,
    "syn_train": featuresForward_synTrain,
    "syn_test": featuresForward_synTest,
    "Combined_train": pd.concat([featuresForward_Train,featuresForward_synTrain], ignore_index=True),
    "Combined_test": pd.concat([featuresForward_Test,featuresForward_synTest], ignore_index=True),
    "Combined_train_original": pd.concat([featuresForward_Train[:17],featuresForward_synTrain[:17]], ignore_index=True),
    "Combined_test_original": pd.concat([featuresForward_Test,featuresForward_Train[-2:]], ignore_index=True)
    }

    # For SVM
    print('\nInitializing SVM classification with MDI features\n'.center(15))
    svm_sf_acc, svm_sf_auroc = evaluate_classifier(support_vector_machine, feature_sets, labels, scenarios)

    # For RF
    print('\nInitializing RF classification with MDI features\n'.center(15))
    rf_sf_acc, rf_sf_auroc = evaluate_classifier(random_forest, feature_sets, labels, scenarios)

    # For MLP
    print('\nInitializing MLP classification with MDI features\n'.center(15))
    mlp_sf_acc, mlp_sf_auroc = evaluate_classifier(multi_layer_perceptron, feature_sets, labels, scenarios)    
    print('\n..........Done\n'.center(30))
    # Construction of the matrix to store the values of classifications
    classification_matrix = pd.DataFrame({
    'SVM MI ACC': svm_mi_acc,
    'SVM MI AUROC': svm_mi_auroc,
    'RF MI ACC': rf_mi_acc,
    'RF MI AUROC': rf_mi_auroc,
    'MLP MI ACC': mlp_mi_acc,
    'MLP MI AUROC': mlp_mi_auroc,
    'SVM MDI ACC': svm_mdi_acc,
    'SVM MDI AUROC': svm_mdi_auroc,
    'RF MDI ACC': rf_mdi_acc,
    'RF MDI AUROC': rf_mdi_auroc,
    'MLP MDI ACC': mlp_mdi_acc,
    'MLP MDI AUROC': mlp_mdi_auroc,    
    'SVM SF ACC': svm_sf_acc,
    'SVM SF AUROC': svm_sf_auroc,
    'RF SF ACC': rf_sf_acc,
    'RF SF AUROC': rf_sf_auroc,
    'MLP SF ACC': mlp_sf_acc,
    'MLP SF AUROC': mlp_sf_auroc
    }, index=['Real-Real', 'Real-Synthetic', 'Synthetic-Real', 'Synthetic-Synthetic', 'Combined', 'Combined_original'])
    print(classification_matrix)
    classification_matrix.to_csv('results/' + args.filename + '_results.csv')
    print('\n..........Classification models done\n'.center(30))
    # Grid search on MDI features only


