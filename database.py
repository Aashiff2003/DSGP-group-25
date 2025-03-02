from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# PostgreSQL Connection URI
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:aashiff12190@localhost/FalconEye"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# User Model
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(), unique=True, nullable=False)
    password_hash = db.Column(db.String(), nullable=False)

    def __repr__(self):
        return f"{self.username}"

# Bird Strike Model
class BirdStrike(db.Model):
    __tablename__ = 'bird_strike'
    id = db.Column(db.Integer, primary_key=True)
    weather = db.Column(db.String(), nullable=False)
    bird_size = db.Column(db.String(), nullable=False)
    bird_species = db.Column(db.String(), nullable=False)
    bird_quantity = db.Column(db.Integer, nullable=False)
    alert_level = db.Column(db.String(), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f"{self.weather} - {self.bird_size} - {self.bird_species} - {self.bird_quantity} - {self.alert_level}"

# Create Tables
def create_database():
    try:
        with app.app_context():
            db.create_all()
        print("Database and tables are ready.")
    except SQLAlchemyError as e:
        print(f"Error creating database: {e}")

# Add New User with Hashed Password
def add_user(username, password):
    try:
        hashed_password = generate_password_hash(password)  # Hash the password
        user = User(username=username, password_hash=hashed_password)
        db.session.add(user)
        db.session.commit()
        return {"message": "User created successfully"}
    except SQLAlchemyError as e:
        db.session.rollback()
        return {"error": str(e)}

# Check Password (Using Hashed Password)
def password_check(username, password):
    try:
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            return {"message": "Login successful"}
        return {"error": "Invalid username or password"}
    except SQLAlchemyError as e:
        return {"error": str(e)}

# Insert Bird Strike Record
def add_bird_strike(weather, bird_size, bird_species, bird_quantity, alert_level):
    try:
        bird_strike = BirdStrike(
            weather=weather,
            bird_size=bird_size,
            bird_species=bird_species,
            bird_quantity=bird_quantity,
            alert_level=alert_level
        )
        db.session.add(bird_strike)
        db.session.commit()
        return {"message": "Bird strike record added successfully"}
    except SQLAlchemyError as e:
        db.session.rollback()
        return {"error": str(e)}

# Fetch All Bird Strike Records
def fetch_records():
    try:
        records = BirdStrike.query.order_by(BirdStrike.timestamp.desc()).all()
        return jsonify([
            {
                "id": record.id,
                "weather": record.weather,
                "bird_size": record.bird_size,
                "bird_species": record.bird_species,
                "bird_quantity": record.bird_quantity,
                "alert_level": record.alert_level,
                "timestamp": record.timestamp
            }
            for record in records
        ])
    except SQLAlchemyError as e:
        return {"error": str(e)}

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
    return fetch_records()

# Run Flask App
if __name__ == "__main__":
    create_database()  
    app.run(debug=True)
