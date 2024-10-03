import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Concatenate
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.backend import random_normal
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

class CCVAE:
    def __init__(self, input_dim, label_dim=2, latent_dim=5, intermediate_dim=512, batch_size=4, epochs=1000):
        """
            Atributes of the C-VAE
            input_dim -> Number of features
            label_dim -> Number of classes
            latent_dim -> Latent dimention reduction
            intermediate_di,m -> Number of neurons in the middle layers of the encoder
            batch_size -> size of the batch during training
            epochs -> Number of epochs during training
        """
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.cc_vae, self.encoder, self.generator = self.create_cc_vae()
    
    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = random_normal(shape=(tf.shape(z_mean)[0], self.latent_dim), mean=0., stddev=0.01)
        return z_mean + tf.math.exp(z_log_sigma) * epsilon
    
    def create_cc_vae(self):
        # Inputs
        features_input = Input(shape=(self.input_dim,), name='features_input')
        label_input = Input(shape=(self.label_dim,), name='label_input')
        
        # Encoder
        encoder_concat = Concatenate()([features_input, label_input])
        h = Dense(self.intermediate_dim, activation='relu')(encoder_concat)
        z_mean = Dense(self.latent_dim)(h)
        z_log_sigma = Dense(self.latent_dim)(h)

        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_log_sigma])

        # Decoder
        decoder_h = Dense(self.intermediate_dim, activation='relu')
        decoder_mean = Dense(self.input_dim, activation='sigmoid')
        h_decoded = decoder_h(Concatenate()([z, label_input]))
        x_decoded_mean = decoder_mean(h_decoded)

        # CC-VAE model
        cc_vae = Model([features_input, label_input], x_decoded_mean)

        # VAE loss
        xent_loss = self.input_dim * binary_crossentropy(features_input, x_decoded_mean)
        kl_loss = - 0.5 * tf.reduce_sum(1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma), axis=-1)
        vae_loss = tf.reduce_mean(xent_loss + kl_loss)

        cc_vae.add_loss(vae_loss)
        cc_vae.compile(optimizer='rmsprop')

        # Encoder Model
        encoder = Model([features_input, label_input], z_mean)

        # Generator Model
        decoder_input = Input(shape=(self.latent_dim,))
        generator_label_input = Input(shape=(self.label_dim,))
        _h_decoded = decoder_h(Concatenate()([decoder_input, generator_label_input]))
        _x_decoded_mean = decoder_mean(_h_decoded)
        generator = Model([decoder_input, generator_label_input], _x_decoded_mean)
        return cc_vae, encoder, generator

    def train(self, x_train, y_train, x_val=None, y_val=None, patience=1000):
        """
            Function to train the VAE with early stop
            x_train -> features to train the VAE
            y_train -> labels to train the VAE
            x_val -> features to validate the vae
            y_vae -> labels to train the vae
            patience -> early stop condition to stop if after N epochs theres no improvement
        """
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]
        if x_val is None or y_val is None:
            self.cc_vae.fit([x_train, y_train], epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks)
        else:
            self.cc_vae.fit([x_train, y_train], epochs=self.epochs, batch_size=self.batch_size, 
                            validation_data=([x_val, y_val], None), callbacks=callbacks)    
            
    def generate_samples(self, latent_samples, labels):
        """
            Function to generate synthetic data
        """
        return self.generator.predict([latent_samples, labels])