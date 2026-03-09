## Problem Formulation

### Goal
The goal of this project is to **predict the next day’s PM2.5 concentration** for a given city using historical air pollution data, weather conditions, and time-based patterns.  
This setup mirrors a real-world forecasting scenario where only past and present information is available.

---

### Problem Type: Supervised Regression
This is a **supervised learning** problem because the model is trained using historical data where the correct output (PM2.5 values) is already known.

It is a **regression task** since PM2.5 is a continuous numerical variable, not a category or label.  
The model learns a function that maps input features to a numerical pollution value.

---

### Target Variable: PM2.5(t+1)
The target variable is **PM2.5(t+1)**, which represents the PM2.5 concentration on the next day.

- `t` refers to the current day  
- `t+1` refers to the following day  

For each row in the dataset, the model uses information available up to day `t` and predicts the pollution level for day `t+1`.  
This formulation avoids using future information and reflects a realistic prediction setting.

---

### Input Features
The model uses three broad categories of input features:

#### 1. Past Pollution Data
Air pollution exhibits strong temporal dependence. High pollution levels on one day often influence levels on subsequent days.  
Examples include:
- PM2.5 values from previous days
- Rolling averages over recent days

#### 2. Weather Features
Weather conditions play a critical role in pollutant dispersion and accumulation.  
Examples include:
- Temperature
- Humidity
- Wind speed
- Rainfall

These features help explain why pollution levels rise or fall rather than relying only on historical trends.

#### 3. Time-Based Features
Time features capture seasonal and human-activity patterns.  
Examples include:
- Day of week
- Month
- Season

Including time-based information allows the model to learn recurring patterns such as winter pollution spikes.

---

### Train–Test Split Strategy
A **time-based split** is used instead of a random split.

- Training data consists of earlier time periods
- Test data consists of later, unseen time periods

This approach prevents data leakage and ensures that the model is evaluated on future data, closely simulating real-world deployment conditions.