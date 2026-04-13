# 🏡 House Price Prediction API

## 📌 Overview

This is a comprehensive Machine Learning API built with **FastAPI** and **Streamlit** for predicting house prices. The project provides a scalable, modular architecture for training ML models, evaluating their performance, and making real-time predictions.

## 🚀 Features

- ✅ **FastAPI Backend** - High-performance REST API with automatic documentation
- ✅ **ML Model Training** - Train models on custom datasets via `/train` endpoint
- ✅ **Batch Predictions** - Support for single and batch predictions
- ✅ **Model Evaluation** - Evaluate model performance with metrics (MAE, RMSE, R²)
- ✅ **Streamlit UI** - Interactive web interface for easy predictions
- ✅ **CSV Upload** - Upload training datasets directly via API
- ✅ **Docker Ready** - Pre-configured Dockerfile for containerized deployment
- ✅ **CORS Enabled** - Cross-origin requests supported
- ✅ **Automatic Documentation** - Swagger UI and ReDoc included

## 📁 Project Structure

```
house-price-ml-api/
├── app/
│   ├── main.py              # FastAPI application with endpoints
│   ├── schemas.py           # Pydantic models for request/response
│   ├── __init__.py
│   └── ml/
│       ├── trainer.py       # Model training and evaluation
│       └── predictor.py     # Model loading and prediction
├── streamlit_app/
│   └── ui.py                # Streamlit interactive UI
├── data/                    # Training data directory
├── models/                  # Saved trained models
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🛠 Technologies Used

- **Python 3.10+** - Programming language
- **FastAPI** - High-performance web framework
- **Uvicorn** - ASGI server
- **Streamlit** - Interactive web app framework
- **scikit-learn** - Machine learning library
- **pandas & NumPy** - Data manipulation
- **Pydantic** - Data validation
- **Docker** - Containerization
- **joblib** - Model serialization

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bikeeprajapati/house-price-ml-api.git
   cd house-price-ml-api
   ```

2. **Create a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Application

### Option 1: FastAPI Backend Only
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Access API at: `http://localhost:8000`

### Option 2: Streamlit UI Only
```bash
streamlit run streamlit_app/ui.py
```
Access UI at: `http://localhost:8501`

### Option 3: Both Together (in separate terminals)

**Terminal 1 - FastAPI:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Streamlit:**
```bash
streamlit run streamlit_app/ui.py
```

## 📚 API Endpoints

### 1. **GET /** - Home
Returns a welcome message.

**Example:**
```bash
curl http://localhost:8000/
```

### 2. **GET /health** - Health Check
Check if the API is running.

**Example:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{"status": "ok"}
```

### 3. **POST /upload-csv** - Upload Training Data
Upload a CSV file for training.

**Example:**
```bash
curl -X POST -F "file=@data.csv" http://localhost:8000/upload-csv
```

### 4. **POST /train** - Train Model
Train the model using uploaded CSV data.

**Example:**
```bash
curl -X POST "http://localhost:8000/train?csv_path=data/data.csv"
```

**Response:**
```json
{
  "train_size": 16512,
  "test_size": 4128,
  "mae": 95000.5,
  "mse": 15000000000,
  "rmse": 122474.0,
  "r2": 0.685,
  "model_path": "models/house_price_model.pkl"
}
```

### 5. **GET /evaluate** - Evaluate Model
Get performance metrics on the full dataset.

**Example:**
```bash
curl http://localhost:8000/evaluate
```

**Response:**
```json
{
  "samples": 21000,
  "mae": 97000.25,
  "mse": 15500000000,
  "rmse": 124500.0,
  "r2": 0.68
}
```

### 6. **POST /predict** - Make Predictions
Predict house prices for single or multiple records.

**Single Record Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "record": {
      "bedrooms": 3,
      "bathrooms": 2,
      "sqft_living": 1500,
      "sqft_lot": 5000,
      "floors": 2,
      "condition": 3,
      "yr_built": 1990
    }
  }'
```

**Batch Records Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft_living": 1500,
        "sqft_lot": 5000,
        "floors": 2,
        "condition": 3,
        "yr_built": 1990
      },
      {
        "bedrooms": 4,
        "bathrooms": 2.5,
        "sqft_living": 2000,
        "sqft_lot": 6000,
        "floors": 2,
        "condition": 4,
        "yr_built": 2000
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [325000.50]
}
```

## 🎨 Streamlit UI Features

The Streamlit interface provides:
- Interactive input fields for house features
- Real-time price prediction
- User-friendly design with emojis
- Error handling and validation

**Input Fields:**
- Bedrooms (number)
- Bathrooms (decimal)
- Living Area (sqft)
- Lot Area (sqft)
- Floors (decimal)
- Condition (1-5 scale)
- Year Built (year)

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t house-price-ml-api .
```

### Run Docker Container
```bash
docker run -p 8000:8000 -p 8501:8501 house-price-ml-api
```

This starts both FastAPI (port 8000) and Streamlit (port 8501).

## 🌐 API Documentation

Once the server is running, access:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 📊 Model Metrics Explained

- **MAE (Mean Absolute Error)** - Average prediction error in dollars
- **MSE (Mean Squared Error)** - Average squared error
- **RMSE (Root Mean Squared Error)** - Square root of MSE
- **R² Score** - Coefficient of determination (0-1, higher is better)

## 🚢 Deployment Options

### Railway
1. Connect your GitHub repository
2. Add environment variables if needed
3. Deploy with one click

### Render
1. Create a new Web Service
2. Select GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

### AWS EC2
1. Launch an EC2 instance with Python 3.10+
2. Clone the repository
3. Install dependencies
4. Run with systemd or supervisor

### Azure App Service
1. Create an App Service
2. Connect to GitHub repository
3. Configure deployment settings
4. Deploy

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
git checkout -b feature/your-feature-name
```
3. **Commit your changes:**
   ```bash
git commit -m "Add your feature description"
```
4. **Push to the branch:**
   ```bash
git push origin feature/your-feature-name
```
5. **Create a Pull Request**

## 📝 License

This project is open-source and available under the MIT License. Feel free to use and modify for personal or commercial projects.

## 💡 Tips for Best Results

1. **Data Quality** - Ensure your training CSV has the same features as expected
2. **Data Size** - More training data = better predictions
3. **Feature Engineering** - Consider preprocessing features for better results
4. **Model Retraining** - Retrain periodically with new data

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Change port for FastAPI
uvicorn app.main:app --port 8001

# Change port for Streamlit
streamlit run streamlit_app/ui.py --server.port 8502
```

### Model Not Found
Make sure to train the model first by calling `/train` or `/upload-csv` before making predictions.

### CSV Upload Issues
Ensure your CSV file has the correct column names matching the model features.

## 📞 Support

For issues, questions, or suggestions:
1. Open an issue on GitHub
2. Check existing issues for solutions
3. Provide detailed error messages and logs

---

**Made with ❤️ by Bikee Prajapati**

Happy Predicting! 🎯