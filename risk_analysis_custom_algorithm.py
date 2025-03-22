import pandas as pd

def calculate_risk_score(row):
    """
    Custom risk prediction algorithm using weighted scores for bird strike incidents.
    """
    # Define risk scores for each factor
    weather_scores = {'No Cloud': 1, 'Some Cloud': 2, 'Overcast': 3}
    size_scores = {'Small': 1, 'Medium': 2, 'Large': 3}
    impact_scores = {
        'Engine Shut Down - Caused damage': 10,
        'Engine Shut Down - No damage': 8,
        'Unknown - Caused damage': 3,
        'Unknown - No damage': 1,
        'Precautionary Landing - Caused damage': 5,
        'Precautionary Landing - No damage': 3,
        'Other - Caused damage': 4,
        'Other - No damage': 2,
        'Aborted Take-off - Caused damage': 10,
        'Aborted Take-off - No damage': 8
    }
    
    # Weight factors
    WEATHER_WEIGHT = 0.2
    SIZE_WEIGHT = 0.3
    IMPACT_WEIGHT = 0.5
    
    # Fetch values
    num_species = row.get('NumberStruckActual', 0)
    size_score = size_scores.get(row.get('WildlifeSize', ''), 0)
    weather_score = weather_scores.get(row.get('ConditionsSky', ''), 0)
    impact_score = impact_scores.get(row.get('MergedEffect', ''), 0)
    
    # Compute weighted risk score
    bird_risk_score = num_species * size_score * SIZE_WEIGHT
    weather_risk_score = weather_score * WEATHER_WEIGHT
    impact_risk_score = impact_score * IMPACT_WEIGHT
    
    total_risk_score = bird_risk_score + weather_risk_score + impact_risk_score
    return total_risk_score

def categorize_risk(alert_score):
    """Categorize risk levels based on the calculated alert score."""
    if alert_score <= 3:
        return 'Low'
    elif alert_score <= 7:
        return 'Moderate'
    else:
        return 'High'

# Example dataframe
data = {
    'NumberStruckActual': [2, 1, 3],
    'WildlifeSize': ['Medium', 'Small', 'Large'],
    'ConditionsSky': ['Some Cloud', 'No Cloud', 'Overcast'],
    'MergedEffect': ['Engine Shut Down - Caused damage', 'Unknown - No damage', 'Precautionary Landing - No damage']
}

df = pd.DataFrame(data)
df['AlertScore'] = df.apply(calculate_risk_score, axis=1)
df['Risk'] = df['AlertScore'].apply(categorize_risk)

print(df)
