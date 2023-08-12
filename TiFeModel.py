import numpy as np
import tensorflow as tf

class SelfAttention(tf.keras.Model):
    def __init__(self, attn_latent_dim, input_shape, **kwargs):
        super().__init__(**kwargs)
        self.time_attn = tf.keras.layers.Attention()
        self.feat_attn = tf.keras.layers.Attention()
        self.Wqt = tf.keras.layers.Dense(attn_latent_dim)
        self.Wkt = tf.keras.layers.Dense(attn_latent_dim)
        self.Wvt = tf.keras.layers.Dense(attn_latent_dim)
        self.Wqf = tf.keras.layers.Dense(attn_latent_dim)
        self.Wkf = tf.keras.layers.Dense(attn_latent_dim)
        self.Wvf = tf.keras.layers.Dense(attn_latent_dim)
        self.concat = tf.keras.layers.Concatenate(axis=-2)
        self.time_extr = tf.keras.layers.Dense(input_shape[0],activation='relu')
        self.feat_extr = tf.keras.layers.Dense(input_shape[1],activation='relu')
    def call(self, x, return_scores=False):
        # Assuming v is k
        qt,kt,vt = self.Wqt(x),self.Wkt(x),self.Wvt(x)
        x = tf.transpose(x,perm=[0,2,1])
        qf,kf,vf = self.Wqf(x),self.Wkf(x),self.Wvf(x)
        if return_scores:
            ta,ts = self.time_attn([qt, kt, vt], return_attention_scores=return_scores)
            fa,fs = self.feat_attn([qf, kf, vf], return_attention_scores=return_scores)
        else:
            ta = self.time_attn([qt, kt, vt], return_attention_scores=return_scores)
            fa = self.feat_attn([qf, kf, vf], return_attention_scores=return_scores)
            ts = None
            fs = None
        x = self.concat([ta,fa])
        x = tf.transpose(x,perm=[0,2,1])
        x = self.time_extr(x)
        x = tf.transpose(x,perm=[0,2,1])
        x = self.feat_extr(x)
        return x,ts,fs
    
class Detector(tf.keras.Model):
    def __init__(self, latent_dim, input_shape, attn_latent_dim, **kwargs):
        super().__init__(**kwargs)
        n_features = input_shape[1]
        self.en_self_attn = SelfAttention(attn_latent_dim, input_shape)
        self.encoder_dense = tf.keras.layers.Dense(latent_dim,
                                                  activation='relu')
        self.decoder_dense = tf.keras.layers.Dense(n_features,
                                                  activation='relu')
        self.de_self_attn = SelfAttention(attn_latent_dim, input_shape)
        
        self.n_features = n_features
        self.latent_dim = latent_dim
        
        
        self.en_sat_scores = None
        self.en_saf_scores = None
        self.de_sat_scores = None
        self.de_saf_scores = None
        
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
    
    @property
    def metrics(self):
        return [
            self.loss_tracker
        ]
        
    def call(self,data,return_scores=False):
        x,est,esf = self.en_self_attn(data,return_scores=return_scores)
        x = self.encoder_dense(x)
        x = self.decoder_dense(x)
        x,dst,dsf = self.de_self_attn(x,return_scores=return_scores)
        
        self.en_sat_scores = est
        self.en_saf_scores = esf
        self.de_sat_scores = dst
        self.de_saf_scores = dsf
        
        return x        
    
    def predict_step(self, data,return_scores=False):
        x,est,esf = self.en_self_attn(data,return_scores=return_scores)
        x = self.encoder_dense(x)
        x = self.decoder_dense(x)
        x,dst,dsf = self.de_self_attn(x,return_scores=return_scores)
        
        self.en_sat_scores = est
        self.en_saf_scores = esf
        self.de_sat_scores = dst
        self.de_saf_scores = dsf
        
        return x  
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)
            loss = tf.reduce_mean(tf.square(y_pred-data))
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {
            'loss': self.loss_tracker.result()
        }
    
    def test_step(self, data):
        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)
            loss = tf.reduce_mean(tf.square(y_pred-data))
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {
            'loss': self.loss_tracker.result()
        }
    