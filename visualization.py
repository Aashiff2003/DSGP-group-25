import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
from flask import Response
from config import create_app
from database import fetch_records

# Initialize Flask app
app = create_app()

def plot_alert_level_distribution():
    with app.app_context():
        df = fetch_records()
        
        if "error" in df:
            return "Error fetching records", 400
        
        df = pd.DataFrame(df)
        plt.figure(figsize=(7, 5))
        sns.countplot(data=df, x="alert_level", palette="Reds")
        plt.xlabel("Alert Level")
        plt.ylabel("Count")
        plt.title("Frequency of Alert Levels")
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return Response(img.getvalue(), mimetype='image/png')

def plot_bird_quantity_vs_time():
    with app.app_context():
        df = fetch_records()

        if "error" in df:
            return "Error fetching records", 400

        df = pd.DataFrame(df)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").resample("H").sum()  # Hourly resampling
        df.reset_index(inplace=True)

        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x="timestamp", y="bird_quantity", marker="o", color="b")
        plt.xlabel("Time")
        plt.ylabel("Bird Quantity")
        plt.title("Bird Quantity Over Time")
        plt.xticks(rotation=45)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return Response(img.getvalue(), mimetype='image/png')
