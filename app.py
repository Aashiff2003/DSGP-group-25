import tensorflow as tf
import os

# Force TensorFlow to use CPU and float32
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.keras.backend.set_floatx('float32')

# Load the .keras model
model_path = "/Users/akshankumarsen/PycharmProjects/aoi/final_weather_classification_model_fixed.keras"
model = tf.keras.models.load_model(model_path)

print("✅ Model loaded successfully in float32 mode using .keras format!")