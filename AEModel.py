import tensorflow as tf

class DetectorAE(tf.keras.Model):
    def __init__(self, latent_dim, input_shape, attn_latent_dim, **kwargs):
        super().__init__(**kwargs)
        n_features = input_shape[1]
        # self.en_self_attn = SelfAttention(attn_latent_dim, input_shape)
        self.encoder_dense = tf.keras.layers.Dense(latent_dim,
                                                  activation='relu')
        self.decoder_dense = tf.keras.layers.Dense(n_features,
                                                  activation='relu')
        # self.de_self_attn = SelfAttention(attn_latent_dim, input_shape)
        
        self.n_features = n_features
        self.latent_dim = latent_dim
        
        
        # self.en_sat_scores = None
        # self.en_saf_scores = None
        # self.de_sat_scores = None
        # self.de_saf_scores = None
        
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
    
    @property
    def metrics(self):
        return [
            self.loss_tracker
        ]
        
    def call(self,data,return_scores=False):
        # x,est,esf = self.en_self_attn(data,return_scores=return_scores)
        x = self.encoder_dense(data)
        x = self.decoder_dense(x)
        # x,dst,dsf = self.de_self_attn(x,return_scores=return_scores)
        
        # self.en_sat_scores = est
        # self.en_saf_scores = esf
        # self.de_sat_scores = dst
        # self.de_saf_scores = dsf
        
        return x        
    
    def predict_step(self, data,return_scores=False):
        # x,est,esf = self.en_self_attn(data,return_scores=return_scores)
        x = self.encoder_dense(data)
        x = self.decoder_dense(x)
        # x,dst,dsf = self.de_self_attn(x,return_scores=return_scores)
        
        # self.en_sat_scores = est
        # self.en_saf_scores = esf
        # self.de_sat_scores = dst
        # self.de_saf_scores = dsf
        
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
