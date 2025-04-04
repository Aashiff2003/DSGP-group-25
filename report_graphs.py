from flask import Flask, send_file, render_template
import matplotlib.pyplot as plt
import pandas as pd
import io
from datetime import datetime, timedelta
import random

app = Flask(__name__)

# Generate Sample Data
def generate_sample_data():
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(7)][::-1]  # last 7 days

    risk_levels = [random.choice(['Low', 'Medium', 'High']) for _ in dates]
    weather = [random.randint(20, 40) for _ in dates]

    df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in dates],
        'RiskLevel': risk_levels,
        'WeatherMetric': weather
    })

    return df

def plot_risk_graph(df):
    # Map Risk Level to numeric for plotting
    risk_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df['RiskLevelNum'] = df['RiskLevel'].map(risk_map)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Date'], df['RiskLevelNum'], marker='o', linestyle='-', color='purple')
    ax.set_title('Bird Strike Risk Level Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Risk Level')
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Low', 'Medium', 'High'])
    ax.grid(True)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def plot_weather_graph(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Date'], df['WeatherMetric'], marker='o', linestyle='-', color='skyblue')
    ax.set_title('Weather Impact Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weather Metric')
    ax.grid(True)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

@app.route('/')
def index():
    return render_template('graphs.html')

@app.route('/download/risk')
def download_risk_graph():
    df = generate_sample_data()
    img = plot_risk_graph(df)
    return send_file(img, mimetype='image/png', as_attachment=True, download_name='risk_graph.png')

@app.route('/download/weather')
def download_weather_graph():
    df = generate_sample_data()
    img = plot_weather_graph(df)
    return send_file(img, mimetype='image/png', as_attachment=True, download_name='weather_graph.png')

if __name__ == '__main__':
    app.run(debug=True)
