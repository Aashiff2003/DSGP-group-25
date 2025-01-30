Weather Classification & Image Enhancement

Overview

This component of the Bird Detection Project focuses on identifying weather conditions in images and applying appropriate enhancement techniques to improve visibility. Since adverse weather (such as rain, fog, and low-light conditions) can degrade image quality, this model ensures that images are processed for optimal bird detection.

Features
	•	Weather Classification: Categorizes images based on weather conditions (e.g., sunny, cloudy, rainy, foggy).
	•	Image Enhancement: Applies adaptive techniques to enhance image clarity for better bird detection.
	•	Fourier Transform for Rain Reduction: Removes high-frequency noise caused by rain to smooth out images.
	•	Contrast and Brightness Adjustments: Enhances visibility in low-light or foggy conditions.

Workflow
	1.	Preprocessing
	•	Load and normalize input images.
	•	Extract relevant features for weather classification.
	2.	Weather Classification Model
	•	Train and test a deep learning or traditional machine learning model to classify weather conditions.
	•	Use labeled datasets containing images of various weather types.
	3.	Image Enhancement Based on Weather
	•	Rainy Conditions: Apply Fourier Transform to smooth out raindrops.
	•	Foggy Conditions: Use contrast stretching or dehazing techniques.
	•	Low-Light Conditions: Adjust brightness and contrast dynamically.
	4.	Output
	•	Generate an enhanced image that improves bird detection accuracy.

Dependencies
	•	Python (>=3.8)
	•	OpenCV
	•	NumPy
	•	TensorFlow / PyTorch (if using deep learning for classification)
	•	Scikit-learn (if using traditional ML methods)

Installation

pip install opencv-python numpy tensorflow scikit-learn

Usage

from weather_classifier import classify_weather
from image_enhancer import enhance_image

# Load image
image = cv2.imread("input.jpg")

# Identify weather conditions
weather = classify_weather(image)

# Enhance image based on weather conditions
enhanced_image = enhance_image(image, weather)

# Save or display the enhanced image
cv2.imwrite("enhanced.jpg", enhanced_image)
cv2.imshow("Enhanced Image", enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

Future Improvements
	•	Fine-tune weather classification using more diverse datasets.
	•	Integrate real-time processing for live video input.
	•	Implement deep learning-based denoising techniques for enhanced clarity.

Contributors
	•	Akshan Kumarasan – Weather Classification & Image Enhancement
Dataset Link- https://www.kaggle.com/datasets/vijaygiitk/multiclass-weather-dataset
