from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("final_model.joblib")

# Load dataset for dropdown values
df = pd.read_csv("new_clean.csv")

# Get unique options dynamically from dataset
CROPS = sorted(df["Crop"].dropna().unique().tolist())
SEASONS = sorted(df["Season"].dropna().unique().tolist())
STATES = sorted(df["State"].dropna().unique().tolist())
SOIL_TYPES = sorted(df["Soil_Type"].dropna().unique().tolist())

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    total_expense = None
    if request.method == "POST":
        crop = request.form['crop']
        season = request.form['season']
        state = request.form['state']
        soil_type = request.form['soil_type']
        area = float(request.form['area'])

        input_df = pd.DataFrame([{
            'Crop': crop,
            'Season': season,
            'State': state,
            'Soil_Type': soil_type,
            'Area': area
        }])

        try:
            cost_per_hectare = model.predict(input_df)[0]
            cost_per_hectare = round(cost_per_hectare, 2)
            total_expense = round(cost_per_hectare * area, 2)
            prediction = cost_per_hectare
        except Exception as e:
            print("Prediction failed:", e)
            prediction = None
            total_expense = None

    return render_template("index.html",
                           crops=CROPS,
                           seasons=SEASONS,
                           states=STATES,
                           soils=SOIL_TYPES,
                           prediction=prediction,
                           total_expense=total_expense)

if __name__ == "__main__":
    app.run(debug=True)
