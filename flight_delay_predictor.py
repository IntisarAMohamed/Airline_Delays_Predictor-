#import all relevent dependencies 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import mongo_connector  

class FlightDelayPredictor:
    def __init__(self, data_path, mongodb_connection_string):
        #Initialize the FlightDelayPredictor with data source and MongoDB connection string.
        self.data_path = data_path
        self.mongodb_connection_string = mongodb_connection_string
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_data_from_mongodb(self):
        # Use the MongoDB connector to load data
        data = mongo_connector.load_data_from_mongodb(self.mongodb_connection_string)
        self.df = pd.DataFrame(data)

    def load_data(self):
        # Load data from a CSV file (if needed)
        self.df = pd.read_csv(self.data_path)

    def prepare_features(self):
        # Select the numeric columns
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns

        # Handle categorical features using one-hot encoding
        categorical_columns = ['Airport Name', 'Airport Country Code', 'Airport Continent']
        encoded_features = pd.get_dummies(self.df[categorical_columns], columns=categorical_columns)

        # Concatenate numeric and encoded categorical features
        self.X = pd.concat([self.df[numeric_columns], encoded_features], axis=1)

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def prepare_target(self):
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(self.df['Flight Status'])
        self.label_encoder = label_encoder

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def train_decision_tree_model(self):
        self.model = DecisionTreeClassifier()
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        #evaluate the trained model
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy}")
        print(classification_report(self.y_test, y_pred))

    def plot_decision_tree(self):
        #plot the decision tree used for classification
        plt.figure(figsize=(15, 10))
        plot_tree(self.model, filled=True, feature_names=self.X.columns)
        plt.show()

if __name__ == "__main__":
    data_path = './data/AirlineDataset.csv'
    mongodb_connection_string = "mongodb://localhost:27017/"

    
    predictor = FlightDelayPredictor(data_path, mongodb_connection_string)
    predictor.load_data_from_mongodb()
    predictor.prepare_features()
    predictor.prepare_target()
    predictor.split_data()
    predictor.train_decision_tree_model()
    predictor.evaluate_model()
    predictor.plot_decision_tree()
