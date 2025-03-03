from flask import Flask, jsonify
from database import db, create_database, add_user, password_check, add_bird_strike, fetch_records

app = Flask(__name__)

# PostgreSQL Connection URI
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:aashiff12190@localhost/FalconEye"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Database
db.init_app(app)
create_database(app)  # Ensure tables are created

# Flask Routes
@app.route('/')
def home():
    return "Bird Strike Detection System API is Running!"

@app.route('/add_user/<string:username>/<string:password>')
def add_user_route(username, password):
    return add_user(username, password)

@app.route('/login/<string:username>/<string:password>')
def login(username, password):
    return password_check(username, password)

@app.route('/insert/<string:weather>/<string:bird_size>/<string:bird_species>/<int:bird_quantity>/<string:alert_level>')
def insert(weather, bird_size, bird_species, bird_quantity, alert_level):
    return add_bird_strike(weather, bird_size, bird_species, bird_quantity, alert_level)

@app.route('/records')
def records():
    return jsonify(fetch_records())

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
