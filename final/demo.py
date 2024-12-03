import pandas as pd
import tkinter as tk
import lime
import lime.lime_tabular
from tkcalendar import Calendar
from xgboost import XGBRegressor
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load dataset and model
df = pd.read_csv("/Users/kirillsobolev/Documents/GitHub/ML_project/ready_data/winter_2023.csv")
model = XGBRegressor()
model.load_model("/Users/kirillsobolev/Documents/GitHub/ML_project/final/xgb_best.json")
X_train = pd.read_csv("/Users/kirillsobolev/Documents/GitHub/ML_project/ready_data/winter_2021.csv")
# Tinker variables 
day_df = None
predictions = None
current_canvas = None
slider = None

# Columns for dataprep functions
past4h_avg_columns = ['airTemp', 'humidity', 'dewpoint', 'precipitation', 'Friction',
       'Road_Surface_Temperature', 'Water_Film_Surface_mm']

forecast_columns = ['airTemp', 'humidity', 'dewpoint', 'precipitation']


# Data prep functions
def calculate_forecast(df, columns):
    """
    Imitates weather forecast using rolling average for next 2 hours 
    """
    for col in columns:
        df[f"fcst_{col}"] = df[col][::-1].rolling(window=12, min_periods=1).mean()[::-1]
    return df

def calculate_avg_past(df, columns):
    """
    Calculates rolling average for 4 hours backwards. 
    Helps the model to catch the trend better
    """
    for col in columns:
        df[f"past4h_avg_{col}"] = df[col].rolling(window=24, min_periods=1).mean()
    return df

# preprocess data
df = calculate_forecast(df, forecast_columns)
df = calculate_avg_past(df, past4h_avg_columns)
df['ts'] = pd.to_datetime(df['ts'])
X_train = calculate_avg_past(X_train, past4h_avg_columns)
X_train = calculate_forecast(X_train, forecast_columns)
X_train = X_train.drop("ts", axis=1)

print(df.columns, len(df.columns))
print(X_train.columns, len(X_train.columns))
# Lime exaplainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    mode='regression')

# Create Object
root = tk.Tk()
 
# Set geometry
root.geometry("800x800")

# Add Calendar
# Frame to hold the calendar and button
calendar_frame = tk.Frame(root)
calendar_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

data_labels_frame = tk.Frame(root)
data_labels_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nw")
cal = Calendar(calendar_frame, selectmode = 'day',
               year = 2023, month = 12,
               day = 22,
               mindate=df.ts.min(),
               maxdate=df.ts.max(),
               foreground="black")

cal.pack(padx=10, pady=10)

# Define variables for storing selected day dataset and predictions


# Returns a select day dataset
def select_day_df(date, df=df):
    selected_df = df[df['ts'].dt.date == pd.to_datetime(date).date()]
    return selected_df

# Plot
def create_plot(df, predictions): 
    global current_canvas
    # Clear the existing canvas if it exists
    if current_canvas:
        current_canvas.get_tk_widget().destroy()
        current_canvas = None
    
    # the figure that will contain the plot 
    fig = Figure(figsize = (10, 3), 
                 dpi = 100) 
  
    # adding the subplot 
    plot1 = fig.add_subplot(111) 
  
    # plotting the graph 
    plot1.plot(df.ts, df['Friction'], label="Friction", color='b', linestyle=":")
    plot1.plot(df.ts, predictions, label="Predicted", color='r')
    plot1.legend()
    plot1.grid(True)
    # creating the Tkinter canvas 
    # containing the Matplotlib figure 
    current_canvas = FigureCanvasTkAgg(fig, 
                               master = root)
    current_canvas.draw() 
  
    # placing the canvas on the Tkinter window
    current_canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=50)


# Makes predictions for day dataset
def on_confirm():
    global day_df, predictions, slider
    # Get date to parse a day
    date_value = cal.get_date()
    # Save day dataset into variable for making predictions
    day_df = select_day_df(date_value, df)
    # Reset index to make same indeces as in predictions list
    # for proper comparison
    day_df = day_df.reset_index(drop=True)
    # Make predictions, turn them into dataframe and shift to 12 points
    # to compare with original friction values
    predictions = model.predict(day_df.drop("ts", axis=1))
    predictions = pd.DataFrame(predictions, columns=["friction_fcst"])
    predictions = predictions["friction_fcst"].shift(12)
    create_plot(day_df, predictions)

    # Clear existing slider if it exists
    if slider:
        slider.destroy()
        slider = None

    def on_slider_change(val):
        index = int(float(val))
        fcst_row = day_df[["ts", "airTemp", "humidity", "precipitation", 
                      "dewpoint", "Road_Surface_Temperature", 
                      "Water_Film_Surface_mm"]].iloc[index]
        fcst_row_data = tk.Label(data_labels_frame, text=f"Original data: \n{fcst_row}")
        fcst_row_data.grid(row=0, column=0)

        predicted_friction_value = round(float(predictions[index]), 3)
        predicted_friction_data = tk.Label(data_labels_frame, text=f"Predicted friction: \n{predicted_friction_value}")
        predicted_friction_data.grid(row=0, column=1)

        original_friction_value = round(float(day_df.Friction.iloc[index+12]), 3)
        original_friction_data = tk.Label(data_labels_frame, text=f"Original Friction: \n{original_friction_value}")
        original_friction_data.grid(row=0, column=2)
        
        accuracy = round((1 - abs(original_friction_value - predicted_friction_value) / abs(original_friction_value)) * 100, 2)
        accuracy_data = tk.Label(data_labels_frame, text=f"Accuracy: \n{accuracy}%")
        accuracy_data.grid(row=0, column=3)

        exp = explainer.explain_instance(day_df.drop("ts", axis=1).values[index], model.predict)
        # Extract feature importance
        importances = exp.as_list()
        importance_df = pd.DataFrame(importances, columns=['Feature', 'Importance'])
        importance_df_data = tk.Label(data_labels_frame, text=f"Feature Importances using LIME: \n{importance_df}")
        importance_df_data.grid(row=0, column=4)


    slider = tk.Scale(
        calendar_frame,
        from_=day_df.index.min() + 12,
        to=day_df.index.max()-12,
        orient="horizontal",
        label="Select data row",
        command=on_slider_change
    )
    slider.pack(pady=10)
 
# Add Button and Label
tk.Button(calendar_frame, text = "Confirm",
       command = on_confirm).pack()

 
# Execute Tkinter
root.mainloop()

