# CO2 Emission Prediction for Vehicles

This project uses Machine Learning (Linear Regression) to predict CO2 emissions based on vehicle characteristics.

## Project Description
The model was trained on the **CO2 Emission by Vehicles** dataset (Canada). 
The model accuracy (R2 Score) is **0.90+ (90%)**.

### Features Used:
* Engine Size(L) — Engine displacement volume
* Cylinders — Number of cylinders
* Fuel Consumption Comb (L/100 km) — Combined fuel consumption in liters per 100 km
* Fuel Consumption Comb (mpg) — Combined fuel consumption in miles per gallon

## Project Structure
* `train.py` — Model training, data visualization, and saving weights to `model.v5`.
* `predict.py` — Console application for testing predictions.
* `site.py` — Web interface for the project based on Streamlit.
* `requirements.txt` — List of required libraries for the project.
* `model.v5` — Saved trained model.

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. To train the model and view graphs:
   python train.py

3. To launch the website:
   streamlit run site.py
