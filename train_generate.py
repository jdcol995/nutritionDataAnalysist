import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
from keras.utils import to_categorical
from VAE import CCVAE
import numpy as np

def train_Vae(train_x, train_y, test_x, test_y, patience):
    """
        Train the VAE with the specified parameters
        train_x = features for training VAE
        train_y = labels for training
        test_x = validation set for VAE
        test_y = validation set for VAE
        patience = for early stop condition

    """
    input_dim = train_x.shape[1]
    labels_one_hot_tr = to_categorical(train_y, num_classes=2)
    labels_one_hot_tst = to_categorical(test_y, num_classes=2)
    vae_model = CCVAE(input_dim=input_dim, latent_dim=5)
    print('CVAE training started...')
    vae_model.train(train_x, labels_one_hot_tr, test_x, labels_one_hot_tst, patience = patience)
    print('CVAE train finished')
    return vae_model


def generate_data(samples, model):
    """
        This fucntion generates synthetic data given two
        parameters
        samples -> Number of new synthetic observations
        model -> VAE model constructed
    """    
    random_latent_vars = np.random.normal(0, 1, (samples, model.latent_dim))
    labels_to_generate = np.array([[1, 0]] * int(samples/2) + [[0, 1]] * int(samples/2))
    synthetic_samples = model.generate_samples(random_latent_vars, labels_to_generate)
    return synthetic_samples

    


