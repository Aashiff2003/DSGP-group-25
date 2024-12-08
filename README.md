FalconEye: Real-Time Bird Detection and Tracking System for Aviation Safety

FalconEye is an advanced bird detection and tracking system designed to enhance aviation safety by minimizing bird strikes near airports. It leverages cutting-edge machine learning technologies like YOLO v8 and integrates real-time weather analysis to adapt detection techniques under varying environmental conditions.

Table of Contents

	•	Features
	•	System Architecture
	•	Installation
	•	Usage
	•	Technologies Used
	•	Requirements
	•	Contributing
	•	License

Features

	•	Real-Time Bird Detection:
	•	Processes live video feeds or uploaded videos to detect bird species and activity.
	•	Weather Adaptability:
	•	Adjusts detection settings dynamically based on real-time weather data.
	•	Risk Scoring:
	•	Provides a calculated risk score and visualizes bird strike likelihood based on detection data.
	•	Alert Mechanisms:
	•	Notifies users with visual and optional audio alerts for immediate decision-making.
	•	User-Friendly Interface:
	•	Web application with a simple, intuitive interface for non-technical users.

System Architecture

	1.	Front-End:
	•	User interface for uploading videos, viewing results, and managing settings.
	2.	Back-End:
	•	Processes video inputs using the YOLO v8 detection model.
	•	Retrieves real-time weather data from external APIs to enhance detection accuracy.
	3.	Database and Storage:
	•	Stores video files, detection results, and metadata.
	4.	Cloud Integration:
	•	Supports deployment on platforms like AWS or Google Cloud.

Installation

1. Clone the Repository

git clone https://github.com/your-repo/falconeye.git
cd falconeye

2. Set Up a Virtual Environment

python -m venv env
source env/bin/activate # Linux/macOS
env\Scripts\activate    # Windows

3. Install Dependencies

pip install -r requirements.txt

4. Configure Environment Variables

Create a .env file with the following keys:

WEATHER_API_KEY=your_weather_api_key
DATABASE_URL=your_database_url
SECRET_KEY=your_secret_key

5. Run the Application

python main.py

Access the application at http://localhost:8000.

Usage

	1.	Sign Up and Log In:
	•	Create an account to access the system.
	2.	Upload a Video:
	•	Upload a video file for bird detection or start a live feed.
	3.	View Results:
	•	Get detailed detection results, including bird species, locations, and risk scores.
	4.	Receive Notifications:
	•	Enable notifications for critical detections or completed processes.

Technologies Used

	•	Programming Language: Python 3.x
	•	Frameworks:
	•	Back-End: FastAPI/Flask
	•	Front-End: React/HTML & CSS
	•	Machine Learning:
	•	YOLO v8 for bird detection.
	•	Custom weather models for condition analysis.
	•	Database: PostgreSQL/MongoDB
	•	Storage: AWS S3/Google Cloud Storage
	•	APIs:
	•	OpenWeatherMap for real-time weather data.

Requirements

	•	Python: 3.8 or higher
	•	Hardware:
	•	GPU (recommended for faster model inference).
	•	At least 8GB RAM for optimal performance.
	•	Dependencies:
	•	Listed in requirements.txt.

Contributing

We welcome contributions! To get started:
	1.	Fork the repository.
	2.	Create a feature branch.
	3.	Submit a pull request with detailed information about your changes.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

For any questions or feedback, contact the project team:
	•	Akshan Kumarasan: akshan.20222376@iit.ac.lk
	•	Aashiff Mohamed: aashiff.20232471@iit.ac.lk
	•	Geethmi Rajapakshe: gethmin.20232272@iit.ac.lk
	•	Shevon Fernando: gethmin.20232272@iit.ac.lk
