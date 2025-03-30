from db import db
from sqlalchemy.exc import SQLAlchemyError
from flask import current_app
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# User Model
class User(db.Model):
    __tablename__ = 'User'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f"<User {self.username}>"

# Report Model
class Report(db.Model):
    __tablename__ = 'report'
    id = db.Column(db.Integer, primary_key=True)
    weather = db.Column(db.String(255), nullable=False)
    bird_size = db.Column(db.String(255), nullable=False)
    bird_species = db.Column(db.String(255), nullable=False)
    bird_quantity = db.Column(db.Integer, nullable=False)
    alert_level = db.Column(db.String(255), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Report {self.weather} | {self.bird_species} x{self.bird_quantity}>"

    def to_dict(self):
        return {
            "id": self.id,
            "weather": self.weather,
            "bird_size": self.bird_size,
            "bird_species": self.bird_species,
            "bird_quantity": self.bird_quantity,
            "alert_level": self.alert_level,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "timestamp": self.timestamp
        }

# Create tables
def create_database(app):
    with app.app_context():
        try:
            db.create_all()
            print("Database and tables are ready.")
        except SQLAlchemyError as e:
            print(f"Error creating database: {e}")

# Add new user
def add_user(username, password):
    try:
        hashed_password = generate_password_hash(password)
        user = User(username=username, password_hash=hashed_password)
        db.session.add(user)
        db.session.commit()
        return {"message": "User created successfully"}
    except SQLAlchemyError as e:
        db.session.rollback()
        return {"error": str(e)}

# Check login credentials
def password_check(username, password):
    try:
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            return {"message": "Login successful"}
        return {"error": "Invalid username or password"}
    except SQLAlchemyError as e:
        return {"error": str(e)}

# Add new report
def add_report(weather, bird_size, bird_species, bird_quantity, alert_level, start_time=None, end_time=None):
    try:
        report = Report(
            weather=weather,
            bird_size=bird_size,
            bird_species=bird_species,
            bird_quantity=bird_quantity,
            alert_level=alert_level,
            start_time=start_time or datetime.utcnow(),
            end_time=end_time
        )
        db.session.add(report)
        db.session.commit()
        return {"message": "Report added successfully"}
    except SQLAlchemyError as e:
        db.session.rollback()
        return {"error": str(e)}

# Fetch all reports
def fetch_reports():
    try:
        with current_app.app_context():
            reports = Report.query.order_by(Report.timestamp.desc()).all()
            return [r.to_dict() for r in reports]
    except SQLAlchemyError as e:
        return {"error": str(e)}