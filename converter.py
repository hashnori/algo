# Example code to convert Keras model to TFLite
import tensorflow as tf

model = tf.keras.models.load_model('product_classifier_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('product_classifier_model.tflite', 'wb') as f:
    f.write(tflite_model)