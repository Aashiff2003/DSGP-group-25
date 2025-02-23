import mysql.connector
from mysql.connector import Error

# Database connection settings
DB_CONFIG = {
    "host": "localhost",
    "user": "root",  # Change if you set a MySQL password
    "password": "",  # Leave empty if using default XAMPP settings
}

# Function to connect to MySQL and create the database if it doesn't exist
def create_database():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS bird_strike_detection")
        print("Database 'bird_strike_detection' is ready.")
    except Error as e:
        print(f"Error creating database: {e}")
    finally:
        cursor.close()
        conn.close()

# Function to connect to the created database and create the table if it doesn't exist
def create_table():
    try:
        conn = mysql.connector.connect(**DB_CONFIG, database="bird_strike_detection")
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bird_detection_records (
                id INT PRIMARY KEY AUTO_INCREMENT,
                num_birds INT NOT NULL,
                weather VARCHAR(50) NOT NULL,
                bird_size VARCHAR(50) NOT NULL,
                alert_level VARCHAR(10) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("Table 'bird_detection_records' is ready.")
    except Error as e:
        print(f"Error creating table: {e}")
    finally:
        cursor.close()
        conn.close()

# Function to insert a record into the table
def insert_record(num_birds, weather, bird_size, alert_level):
    try:
        conn = mysql.connector.connect(**DB_CONFIG, database="bird_strike_detection")
        cursor = conn.cursor()
        sql = "INSERT INTO bird_detection_records (num_birds, weather, bird_size, alert_level) VALUES (%s, %s, %s, %s)"
        values = (num_birds, weather, bird_size, alert_level)
        cursor.execute(sql, values)
        conn.commit()
        print("Record inserted successfully.")
    except Error as e:
        print(f"Error inserting record: {e}")
    finally:
        cursor.close()
        conn.close()

# Function to fetch all records from the table
def fetch_records():
    try:
        conn = mysql.connector.connect(**DB_CONFIG, database="bird_strike_detection")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM bird_detection_records ORDER BY timestamp DESC")
        records = cursor.fetchall()
        if records:
            print("\n--- Bird Strike Detection Records ---")
            for record in records:
                print(record)
        else:
            print("No records found.")
    except Error as e:
        print(f"Error fetching records: {e}")
    finally:
        cursor.close()
        conn.close()

# Main execution
if __name__ == "__main__":
    create_database()  # Ensure database exists
    create_table()  # Ensure table exists
    insert_record(5, "Cloudy", "Medium", "Moderate")  # Insert a sample record
    fetch_records()  # Retrieve and display records
