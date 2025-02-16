from pymongo import MongoClient

# Connect to MongoDB (using localhost:27017 as the default URI)
client = MongoClient("mongodb://localhost:27017/")
db = client["autism_db"]  # The database name
