import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
from flask import Response
from database import fetch_data



# Function to generate plot
def plot_bird_quantity_vs_weather():
    df = fetch_data()
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=df['weather'], y=df['bird_quantity'], estimator=sum, palette="Blues")
    plt.xlabel("Weather Condition")
    plt.ylabel("Total Bird Quantity")
    plt.title("Bird Quantity in Different Weather Conditions")

    # Convert plot to image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return Response(img.getvalue(), mimetype='image/png')

# Function to generate Alert Level Frequency Plot
def plot_alert_level_distribution():
    df = fetch_data()
    
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="alert_level", palette="Reds")
    plt.xlabel("Alert Level")
    plt.ylabel("Count")
    plt.title("Frequency of Alert Levels")

    # Convert plot to image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return Response(img.getvalue(), mimetype='image/png')
