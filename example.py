import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


model = load_model('product_classifier_model.keras')

# Example image file path
image_path = 'check'

# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize pixel values

# Make predictions
predictions = model.predict(img_array)

# Post-process predictions (binary classification)
predicted_class = 1 if predictions[0][0] > 0.5 else 0
print("Predicted Class:", predicted_class)
