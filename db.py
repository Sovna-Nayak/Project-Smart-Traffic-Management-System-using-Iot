from pymongo import MongoClient

def get_db():
    # Local MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    
    # For MongoDB Atlas (example):
    # client = MongoClient("mongodb+srv://<username>:<password>@cluster.mongodb.net/")

    db = client['traffic_system']
    return db


