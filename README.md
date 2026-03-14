# Metal Coil Defect Detection (Machine Learning Project)

## Overview
This project explores the factors that lead to defects in metal coil production using machine learning techniques. By combining PLC production sensor data with defect inspection records, the goal is to identify key process variables associated with defect occurrence and build predictive models.

## Problem Statement
In industrial manufacturing, defects in metal coils can lead to material waste and increased production costs. Understanding which production parameters contribute to defects can help improve process control and product quality.

This project investigates:
- Which production conditions are associated with defect formation
- Whether machine learning models can predict defect occurrence

## Data
The project uses two main datasets:

**PLC Production Data**
- Sensor measurements recorded during production
- Variables such as temperature, pressure, and production speed
- Measurements collected approximately every few meters of material

**Defect Detection Data**
- Records of defects detected in the coils
- Includes coil identifiers and defect positions along the material

These datasets are aligned to associate production conditions with defect events.

## Methodology
The project follows a typical machine learning workflow:

1. Data understanding and exploration
2. Data cleaning and preprocessing
3. Merging PLC data with defect records
4. Feature exploration and visualization
5. Training machine learning models to predict defects
6. Evaluating model performance and identifying important variables

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook
