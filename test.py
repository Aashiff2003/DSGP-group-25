from tensorflow.keras.models import load_model

# Load the model
model = load_model('models/final_lstm_bird_size_model.h5')

# Display model summary
model.summary()


# Print input details
print("Input details:")
print(model.inputs)

