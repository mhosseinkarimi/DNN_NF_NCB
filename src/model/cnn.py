import numbers

import numpy as np
import tensorflow as tf
from keras.layers import BatchNormalization, Dense, Dropout, Input, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D


class CNN:
    def __init__(self, base_model, FCN, input_shape, output_dim, dropout, loss, model_save_dir=None):
        self.model = self.create_model(base_model, FCN, input_shape, output_dim, dropout)
        self.loss = loss
        self.model_save_dir = model_save_dir
        
    def summary(self):
        self.model.summary()
    
    def create_model(self, base_model, FCN, input_shape, output_dim, dropout):
        input_ = Input(shape=input_shape)
        x = Conv2D(3, (3, 3), padding="same", activation="relu")(input_)
        x = base_model(x)
        for i, units in enumerate(FCN):
            x = Dense(units, activation="relu")(x)
            x = BatchNormalization()(x)
            if dropout:
                if isinstance(dropout, numbers.Number):
                    x = Dropout(rate=dropout)(x)
                else:
                    x = Dropout(rate=0.2)(x)
        output = Dense(output_dim, activation="linear")(x)
        model = Model(inputs=input_, outputs=output)
        return model
    
    def predict(self, data):
        return self.model(data)
    
    def __call__(self, data):
        return self.predict(data)
        
    def train(self,
              train_data, train_label,
              val_data, val_label,
              epochs,
              batch_size=2048,
              optimizer=Adam,
              lr=0.01,
              lr_scheduler=None,
              device="/CPU:0"
              ):

        # initialize optimizer
        optimizer = optimizer(learning_rate=lr)
        
        train_steps_per_epoch = len(train_data) // batch_size
        val_steps_per_epoch = len(val_data) // batch_size
        
        # train and validation metrics and losses at the end of epoch
        train_loss_epochs = []
        val_loss_epochs = []
        train_infer_loss_epochs = []
        
        min_val_loss = 1000
        
        with tf.device(device):
            # training Loop
            for epoch in range(epochs):
                # initialize losses at the begining of the epoch
                epoch_train_loss, epoch_val_loss, epoch_train_infer_loss = 0.0, 0.0, 0.0                
                
                # update lr
                if lr_scheduler:
                    lr = lr_scheduler.update(epoch)
                optimizer.learning_rate.assign(lr)
                
                # training
                with tqdm(total=train_steps_per_epoch, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
                    for batch in range(train_steps_per_epoch):
                        batch_data = train_data[batch*batch_size:(batch+1)*batch_size]
                        batch_label = train_label[batch*batch_size:(batch+1)*batch_size]
                        
                        with tf.GradientTape() as tape:
                            # Forwards pass
                            pred_label = self.model(batch_data, training=True)
                            pred_label = tf.cast(pred_label, dtype=tf.float64)
                            # Loss Calculation
                            train_loss = tf.reduce_mean(self.loss(batch_label, pred_label))
                        # Train loss in inference phase
                        pred_label = self.model(batch_data)
                        pred_label = tf.cast(pred_label, dtype=tf.float64)
                        train_inference_loss = tf.reduce_mean(self.loss(batch_label, pred_label))
                        
                        # Applying the gradients
                        gradients = tape.gradient(train_loss,self. model.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                        epoch_train_infer_loss += train_inference_loss
                        epoch_train_loss  += train_loss
                        
                        pbar.set_postfix(loss=train_loss.numpy())
                        pbar.update(1)
                    
                    # loss on the validation dataset
                    for batch in range(val_steps_per_epoch):
                        batch_data = val_data[batch*batch_size:(batch+1)*batch_size]
                        batch_label = val_label[batch*batch_size:(batch+1)*batch_size] 
                        
                        pred_label = self.model(batch_data)
                        pred_label = tf.cast(pred_label, dtype=tf.float64)
                        val_loss = tf.reduce_mean(self.loss(batch_label, pred_label))                        
                        
                        
                        epoch_val_loss += val_loss
                
                # normalize loss
                epoch_train_loss /= train_steps_per_epoch
                epoch_train_infer_loss /= train_steps_per_epoch
                epoch_val_loss /= val_steps_per_epoch                
                
                # save the loss value at the end of epoch
                train_loss_epochs.append(epoch_train_loss)
                val_loss_epochs.append(epoch_val_loss)
                train_infer_loss_epochs.append(epoch_train_infer_loss)
                
                # Saving the model with the best validation loss
                if (epoch_val_loss < min_val_loss) and self.model_save_dir:
                    self.model.save_weights(self.model_save_dir)
                    min_val_loss = epoch_val_loss

                print(f"Epoch {epoch + 1}, Train CMAE: {epoch_train_loss:.3f}, Test CMAE: {epoch_val_loss:.3f}, Train Inference CMAE: {epoch_train_infer_loss:.3f}, lr: {float(tf.keras.backend.get_value(optimizer.lr))}")
        print(f"Minimum Validation Loss: {np.min(val_loss_epochs)}")
        loss_dict = {
            "train": np.array(train_loss_epochs),
            "train_infer": np.array(train_infer_loss_epochs),
            "val": np.array(val_loss_epochs)
        }
        return loss_dict