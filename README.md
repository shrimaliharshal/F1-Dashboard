# F1-Dashboard
https://f1-insights.streamlit.app/



## Overview

This project is an F1 Analytics Dashboard that provides insights into Formula 1 race data. It leverages the Streamlit framework to create an interactive and visually appealing interface for users to explore various aspects of F1 races.

## Features

- Visualize lap times of all drivers in a particular race.
- Explore team performance through box plots.
- Analyze tire strategies, position changes, and telemetry comparisons.
- Select specific years, Grand Prix events, and drivers for detailed analysis.

## Technologies Used

- Python
- Streamlit
- FastF1 (for accessing Formula 1 data)
- Matplotlib
- Seaborn

Select the desired year, Grand Prix event, and drivers.
Explore different sections of the dashboard for detailed insights.


1. **Functions**:
   - `tyre_strategies(year, gpname)`: This function visualizes the tire strategies of all drivers at a specific Grand Prix for a given year.
   - `laptimes_scatter(year, gpname)`: This function creates a scatter plot of lap times for each driver at a specific Grand Prix in a given year.
   - `telemetry_comparision(year, gp_name, p1, p2)`: This function compares telemetry data between two specific drivers at a particular Grand Prix in a given year.
   - `team_speeds(year, gp_name)`: This function visualizes the speed of different teams at a specific Grand Prix in a given year.
   - `driver_speeds(year, gpname, p1)`: This function generates a speed map for a particular driver at a specific Grand Prix in a given year.
   - `results(year)`: This function retrieves and processes race results data for a specific year in the Formula 1 season.

2. **System Design Elements**:
   - **Streamlit**: The main application is built using Streamlit, a Python library for creating web applications. It provides an interactive dashboard for users to select various analysis options.
   - **FastF1 Library**: The analysis repository utilizes the FastF1 library, which is a Python library used for fetching and analyzing Formula 1 data. It provides functionalities to access telemetry data, lap times, race results, and more.
   - **Matplotlib and Plotly**: Matplotlib and Plotly libraries are used for creating interactive visualizations such as scatter plots, speed maps, telemetry comparisons, and more.
   - **Pandas and NumPy**: These libraries are used for data manipulation and processing race data efficiently.
   - **Ergast API**: The Ergast API is used to retrieve race schedules, results, and other Formula 1 data for analysis.

These functions and system design elements work together to provide users with insightful analysis and visualizations of Formula 1 race data for different Grand Prix events across multiple years.
