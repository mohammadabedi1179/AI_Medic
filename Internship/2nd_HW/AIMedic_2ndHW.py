import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Flatten, Input, Dense
from tensorflow.keras import Model



class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
                print("\nReached 99% accuracy on validation so cancelling training!") 
                self.model.stop_training = True

class Train():
  
  def __init__(self, dataset : str):

    """ load specific dataset from tensorflow_datasets"""
    self.train_data, self.validation_data = tfds.load(dataset, split=['train', 'test'], as_supervised=True, shuffle_files=True)

  def __normalize__(image, label):
    """ 
    scalling pixel values from [0, 255] to [0, 1]
    image : tensor of images
    label : tensor of lables
    """
    image = tf.divide(tf.cast(image, tf.float32), 255.)
    
    return image, label
  
  def __preprocess__(self, batch_size):
    """
    create batches of data and mormalize data
    batch_size : indicates number of examples in each batch
    """

    train_data = self.train_data
    validation_data = self.validation_data

    train_data = train_data.map(Train.__normalize__, num_parallel_calls=tf.data.AUTOTUNE)
    validation_data = validation_data.map(Train.__normalize__, num_parallel_calls=tf.data.AUTOTUNE)
    
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    validation_data = validation_data.batch(batch_size)
    validation_data = validation_data.prefetch(tf.data.AUTOTUNE)

    self.train_data = train_data
    self.validation_data = validation_data
  
  def __create_model__(input_shape):
    
    inputs = Input(shape=input_shape, name='Input_layer')
    x = Flatten(name='Flatten_layer')(inputs)
    x = Dense(32, activation='relu', name='1st_dense_layer')(x)
    x = Dense(16, activation='relu', name='2nd_dense_layer')(x)
    x = Dense(10, activation='softmax', name='3rd_dense_layer')(x)
    model = Model(inputs, x)
    
    return model
  
  def training(self, input_shape : tuple = (28, 28), batch_size : int = 128):
    
    Train.__preprocess__(self, batch_size)
    train_data = self.train_data
    validation_data = self.validation_data
    model = Train.__create_model__(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
  
    call_back = myCallback()
    model.fit(train_data, validation_data=validation_data, epochs=50, callbacks=[call_back])

    return model

if __name__ == '__main__':
  Train('mnist').training((28, 28), 256)