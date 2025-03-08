from flask import Flask
from database import db, create_database

def create_app():
    app = Flask(__name__)

    # PostgreSQL Connection URI
    app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:aashiff12190@localhost/FalconEye"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize Database
    db.init_app(app)
    create_database(app)  # Ensure tables are created

    return app
