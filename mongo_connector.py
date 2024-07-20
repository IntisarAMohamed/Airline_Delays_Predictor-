import pymongo
import csv

# Connect to MongoDB
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["AirlineDatabase"]
mycol = mydb["AirlineCollection"]

# Read data from a CSV file and insert it into the collection
csv_file = './data/AirlineDataset.csv'

with open(csv_file, mode='r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        mycol.insert_one(row)

# Close the MongoDB connection
myclient.close()
