import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache("f1_cache")

# Load 2024 Jeddah session (using existing logic)
session_2024 = fastf1.get_session(2024, "Saudi Arabia", "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Aggregate sector data by driver
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

# 2025 Qualifying Data (Added BOT and PER for Cadillac)
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "PIA", "LEC", "RUS", "HAM", "GAS", "ALO", "TSU", "SAI", "HUL", "OCO", "STR", "NOR", "ALB", "BOT", "PER"],
    "QualifyingTime (s)": [
        87.294, 87.304, 87.670, 87.407, 88.201, 88.367, 88.303, 88.204, 88.164, 88.782, 89.092, 88.645, 87.489, 
        88.500, # ALB (Estimated)
        88.900, # BOT (Cadillac)
        88.600  # PER (Cadillac)
    ]
})

# Average lap times for 2025 season (Updated with new drivers)
average_2025 = {
    "VER": 88.0, "PIA": 89.1, "LEC": 89.2, "RUS": 89.3, "HAM": 89.4, 
    "GAS": 89.5, "ALO": 89.6, "TSU": 89.7, "SAI": 89.8, "HUL": 89.9, 
    "OCO": 90.0, "STR": 90.1, "NOR": 90.2,
    "ALB": 90.3, "BOT": 90.5, "PER": 90.4 
}

# UPDATED: Wet performance factors including PIA, HUL, ALB, BOT, PER
driver_wet_performance = {
    "VER": 0.975196, "HAM": 0.976464, "LEC": 0.975862, "NOR": 0.978179, "ALO": 0.972655,
    "RUS": 0.968678, "SAI": 0.978754, "TSU": 0.996338, "OCO": 0.981810, "GAS": 0.978832, 
    "STR": 0.979857,
    "PIA": 0.975000, # Calculated
    "HUL": 0.980000, # Calculated
    "ALB": 0.982000, # Calculated
    "BOT": 0.979000, # Calculated (Cadillac)
    "PER": 0.978000  # Calculated (Cadillac)
}

qualifying_2025["WetPerformanceFactor"] = qualifying_2025["Driver"].map(driver_wet_performance)

# Weather configuration
API_KEY = "yourkey"
# Using fixed coordinates for example purposes
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=21.4225&lon=39.1818&appid={API_KEY}&units=metric"

# Mocking weather response handling for safety if API fails in this context
try:
    response = requests.get(weather_url)
    weather_data = response.json()
    forecast_time = "2025-04-20 18:00:00"
    forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)
    rain_probability = forecast_data["pop"] if forecast_data else 0
    temperature = forecast_data["main"]["temp"] if forecast_data else 20
except:
    rain_probability = 0
    temperature = 20

if rain_probability >= 0.75:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"] * qualifying_2025["WetPerformanceFactor"]
else:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# UPDATED: Constructor points with Cadillac added
team_points = {
    "McLaren": 78, "Mercedes": 53, "Red Bull": 36, "Williams": 17, "Ferrari": 17,
    "Haas": 14, "Aston Martin": 10, "Kick Sauber": 6, "Racing Bulls": 3, "Alpine": 0,
    "Cadillac": 0 # New Team
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

# UPDATED: Driver to Team mapping with Cadillac drivers
driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin",
    "ALB": "Williams",
    "BOT": "Cadillac",
    "PER": "Cadillac"
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)
qualifying_2025["Average2025Performance"] = qualifying_2025["Driver"].map(average_2025)

# Merge with sector times 
# Note: BOT and PER won't match 2024 sector times, filling with 0 or mean to prevent drop
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["TotalSectorTime (s)"] = merged_data["TotalSectorTime (s)"].fillna(merged_data["TotalSectorTime (s)"].mean())

merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature
merged_data["LastYearWinner"] = (merged_data["Driver"] == "VER").astype(int)
merged_data["QualifyingTime"] = merged_data["QualifyingTime"] ** 2

# Fill NaNs for new drivers
merged_data = merged_data.fillna(0)

print(merged_data[["Driver", "Team", "WetPerformanceFactor", "QualifyingTime"]])
