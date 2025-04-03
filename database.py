from db import db
from sqlalchemy.exc import SQLAlchemyError
from flask import current_app
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# User Model
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f"<User {self.username}>"


# Report (Bird Strike) Model
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


# Report Schedule History Model
class ReportScheduleHistory(db.Model):
    __tablename__ = 'report_schedule_history'
    id = db.Column(db.Integer, primary_key=True)
    schedule_date = db.Column(db.DateTime, nullable=False)
    employee_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(100))
    bird_count = db.Column(db.Integer)
    weather = db.Column(db.String(255))
    visual_graphs = db.Column(db.LargeBinary)

    employee = db.relationship('User', backref='scheduled_reports')


# Report History Model
class ReportHistory(db.Model):
    __tablename__ = 'report_history'
    id = db.Column(db.Integer, primary_key=True)
    report_date = db.Column(db.DateTime, nullable=False)
    employee_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(100))
    bird_count = db.Column(db.Integer)
    weather = db.Column(db.String(255))
    visual_graphs = db.Column(db.LargeBinary)

    employee = db.relationship('User', backref='report_history')


# Create tables
def create_database(app):
    with app.app_context():
        try:
            db.create_all()
            print("Database and tables are ready.")
        except SQLAlchemyError as e:
            current_app.logger.error(f"Error creating database: {e}")


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


# Add new bird report
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


# Fetch all reports (bird strikes)
def fetch_reports():
    try:
        records = Report.query.order_by(Report.timestamp.desc()).all()
        return [record.to_dict() for record in records]
    except RuntimeError:  # no application context
        with current_app.app_context():
            records = Report.query.order_by(Report.timestamp.desc()).all()
            return [record.to_dict() for record in records]
    except SQLAlchemyError as e:
        return {"error": str(e)}


# Add to Report Schedule History
def add_schedule_history(schedule_date, employee_id, title, bird_count, weather, visual_graph_data):
    try:
        record = ReportScheduleHistory(
            schedule_date=schedule_date,
            employee_id=employee_id,
            title=title,
            bird_count=bird_count,
            weather=weather,
            visual_graphs=visual_graph_data
        )
        db.session.add(record)
        db.session.commit()
        return {"message": "Report schedule history added"}
    except SQLAlchemyError as e:
        db.session.rollback()
        return {"error": str(e)}


# Add to Report History
def add_report_history(report_date, employee_id, title, bird_count, weather, visual_graph_data):
    try:
        record = ReportHistory(
            report_date=report_date,
            employee_id=employee_id,
            title=title,
            bird_count=bird_count,
            weather=weather,
            visual_graphs=visual_graph_data
        )
        db.session.add(record)
        db.session.commit()
        return {"message": "Report history added"}
    except SQLAlchemyError as e:
        db.session.rollback()
        return {"error": str(e)}


# Fetch all schedule history
def fetch_schedule_history():
    try:
        records = ReportScheduleHistory.query.order_by(ReportScheduleHistory.schedule_date.desc()).all()
        return [{
            "id": r.id,
            "schedule_date": r.schedule_date,
            "employee_id": r.employee_id,
            "title": r.title,
            "bird_count": r.bird_count,
            "weather": r.weather
        } for r in records]
    except RuntimeError:  # no app context
        with current_app.app_context():
            records = ReportScheduleHistory.query.order_by(ReportScheduleHistory.schedule_date.desc()).all()
            return [{
                "id": r.id,
                "schedule_date": r.schedule_date,
                "employee_id": r.employee_id,
                "title": r.title,
                "bird_count": r.bird_count,
                "weather": r.weather
            } for r in records]
    except SQLAlchemyError as e:
        return {"error": str(e)}


# Fetch all report history
def fetch_report_history():
    try:
        records = ReportHistory.query.order_by(ReportHistory.report_date.desc()).all()
        return [{
            "id": r.id,
            "report_date": r.report_date,
            "employee_id": r.employee_id,
            "title": r.title,
            "bird_count": r.bird_count,
            "weather": r.weather
        } for r in records]
    except RuntimeError:  # no app context
        with current_app.app_context():
            records = ReportHistory.query.order_by(ReportHistory.report_date.desc()).all()
            return [{
                "id": r.id,
                "report_date": r.report_date,
                "employee_id": r.employee_id,
                "title": r.title,
                "bird_count": r.bird_count,
                "weather": r.weather
            } for r in records]
    except SQLAlchemyError as e:
        return {"error": str(e)}