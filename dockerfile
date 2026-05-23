FROM python:3.10-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install packages one by one
RUN pip install --upgrade pip setuptools wheel cython
RUN pip install numpy==1.24.3
RUN pip install pandas==2.0.3 --no-build-isolation
RUN pip install scikit-learn==1.3.0
RUN pip install matplotlib==3.7.2
RUN pip install seaborn==0.12.2
RUN pip install plotly==5.18.0
RUN pip install joblib==1.3.2
RUN pip install xgboost==1.7.6
RUN pip install catboost==1.2.0
RUN pip install lightgbm==4.0.0
RUN pip install gunicorn==21.2.0
RUN pip install streamlit==1.28.1

# Copy application
COPY . .

# Create models
RUN mkdir -p models data && \
    python -c "
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=10)
model.fit(np.random.rand(100,8), np.random.rand(100)*500000)
joblib.dump(model, 'models/best_model.pkl')
"

EXPOSE 8501
CMD streamlit run ui/app_dark_professional.py --server.port=$PORT --server.address=0.0.0.0