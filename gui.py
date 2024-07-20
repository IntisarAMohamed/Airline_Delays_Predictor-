import tkinter as tk
from tkinter import messagebox
import pandas as pd
from flight_delay_predictor import FlightDelayPredictor

# Static Username and Password for Validation
valid_username = "user"
valid_password = "password"

# Create a main application window
app = tk.Tk()
app.title("Flight Delay Predictor")
app.geometry("400x200")

# Create labels and entry widgets for username and password
username_label = tk.Label(app, text="Username")
username_label.pack()
username_entry = tk.Entry(app, width=30)
username_entry.pack()

password_label = tk.Label(app, text="Password")
password_label.pack()
password_entry = tk.Entry(app, show="*", width=30)
password_entry.pack()

# Function to check login credentials
def login():
    username = username_entry.get()
    password = password_entry.get()
# authentication to check username and password entred by user 
    if username == valid_username and password == valid_password:
        app.withdraw()  # Hide the login window
        open_button_form()
    else:
        messagebox.showerror("Login Error", "Invalid username or password")

# Create a login button
login_button = tk.Button(app, text="Login", command=login)
login_button.pack()

# Function to open the button form
def open_button_form():
    #Open the main form with options to view the Decision Tree and flight delays.
    
    button_form = tk.Toplevel(app)
    button_form.title("Button Form")
    button_form.geometry("400x200")

    view_decision_tree_button = tk.Button(button_form, text="View Decision Tree", command=view_decision_tree)
    view_decision_tree_button.pack()

    predict_button = tk.Button(button_form, text="Predict", command=open_main_form)
    predict_button.pack()

predictor = None  

# Function to view the decision tree
#this will open a seperate window to show the decision tree
def view_decision_tree():
    global decision_tree_photo
    decision_tree_window = tk.Toplevel(app)
    decision_tree_window.title("Decision Tree")
    decision_tree_window.geometry("800x600")

    try:
        predictor.plot_decision_tree()  # Plot the decision tree
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to open the main form
def open_main_form():
    # Opens the main form for users to input airport information and make predictions.
    
    main_form = tk.Toplevel(app)
    main_form.title("Main Form")
    main_form.geometry("400x300")

    Airport_Name_Label = tk.Label(main_form, text="Airport Name")
    Airport_Name_Label.pack()
    Airport_Name_box = tk.Entry(main_form, width=40)
    Airport_Name_box.pack()

    Airport_Countrycode_Label = tk.Label(main_form, text="Airport Country Code")
    Airport_Countrycode_Label.pack()
    Airport_Countrycode_box = tk.Entry(main_form, width=40)
    Airport_Countrycode_box.pack()

    Airport_Continent_Label = tk.Label(main_form, text="Airport Continent")
    Airport_Continent_Label.pack()
    Airport_Continent_box = tk.Entry(main_form, width=40)
    Airport_Continent_box.pack()

# Function to submit the form
    def submit_form():
# this function submits the airport information form and make predictions based on user input.
        
        airport_name = Airport_Name_box.get()
        country_code = Airport_Countrycode_box.get()
        airport_continent = Airport_Continent_box.get()

        if predictor is not None:
            input_data = pd.DataFrame({
                'Airport Name': [airport_name],
                'Airport Country Code': [country_code],
                'Airport Continent': [airport_continent]
            })
            prediction = predictor.model.predict(input_data)
            flight_status = predictor.label_encoder.inverse_transform(prediction)[0]

            if flight_status.lower() == "delayed":
                message = "Your flight is Delayed"
            else:
                message = "Your flight is on Time"

            messagebox.showinfo(message)

    # submit button
    submit_button = tk.Button(main_form, text="Submit", command=submit_form)
    submit_button.pack()

if __name__ == "__main__":
    data_path = './data/AirlineDataset.csv'
    mongodb_connection_string = "mongodb://localhost:27017/"

    # Initialize the FlightDelayPredictor
    predictor = FlightDelayPredictor(data_path, mongodb_connection_string)

    # Load data and prepare features/target before running the main app
    predictor.load_data()
    predictor.prepare_features()
    predictor.prepare_target()
    predictor.split_data()
    predictor.train_decision_tree_model()
    predictor.evaluate_model()

    app.mainloop()  # Main application loop at the end
