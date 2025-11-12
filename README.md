# Assignment_3
EMI Prediction 
A complete end-to-end Machine Learning and MLOps project designed to predict EMI eligibility and maximum EMI amount using real-world financial data. The project integrates data science workflows, MLflow experiment tracking, and a Streamlit web application for interactive model deployment

Tech Stack:-
Python, Pandas, NumPy, Scikit-learn, XGBoost
MLflow for experiment tracking & model registry
Streamlit for web interface
Matplotlib, Plotly for visualization
GitHub Actions for automated deployment.

Project Workflow
Step 1: Data Loading & Preprocessing
Process 400K+ financial records across 5 EMI scenarios.
Handle missing values, inconsistencies, and duplicates.
Perform data validation and split datasets for model development.

Step 2: Exploratory Data Analysis (EDA)
Visualize EMI eligibility trends across different lending scenarios.
Explore correlations between financial and demographic variables.
Generate statistical insights and business-driven conclusions.

Step 3: Feature Engineering
Create financial ratios and risk-based features.
Apply categorical encoding, scaling, and interaction features.

Step 4: Machine Learning Model Development
Classification Models: Logistic Regression, Random Forest, XGBoost, and others.
Regression Models: Linear Regression, Random Forest Regressor, XGBoost Regressor.
Evaluate models using accuracy, F1-score, RMSE, and R² metrics.

Step 5: Model Tracking with MLflow
Log model parameters, metrics, and artifacts in MLflow.
Compare experiment runs and register best-performing models for deployment.

Step 6: Streamlit Application Development
Build a multi-page Streamlit web app for real-time EMI predictions.
Integrate interactive visualizations and MLflow dashboard.

Step 7: Cloud Deployment & Production
Deploy on Streamlit Cloud with CI/CD integration via GitHub Actions.
Ensure responsive design, robust error handling, and user feedback mechanisms.
