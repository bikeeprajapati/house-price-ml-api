FastAPI ML Prediction Service

This project is a Machine Learning Prediction API built using FastAPI, designed to provide real-time predictions for single or multiple records.
It includes a modular structure for training, saving, and serving ML models efficiently.

🚀 Features

FastAPI-powered REST API

/predict endpoint supporting:

Single record prediction

Batch prediction

Trained ML model loaded automatically

Clean folder structure for scalable ML projects

Fully asynchronous server

Ready for deployment (Docker / Railway / Render)

Includes .gitignore for Python + VS Code + FastAPI

📁 Project Structure
fastapi-ml-project/
│── app/
│   ├── main.py
│   ├── model.py
│   ├── schema.py
│   ├── __init__.py
│── model/
│   ├── trained_model.pkl
│── data/
│   └── sample_input.json
│── requirements.txt
│── .gitignore
│── README.md

🧪 How to Run Locally
1. Create and Activate Virtual Environment
python -m venv venv


Windows:

venv\Scripts\activate


Linux / Mac:

source venv/bin/activate

2. Install Dependencies
pip install -r requirements.txt

3. Start the FastAPI Server
uvicorn app.main:app --reload

4. Open the API Docs

Swagger UI:
➡️ http://127.0.0.1:8000/docs

Redoc:
➡️ http://127.0.0.1:8000/redoc

📌 Prediction Example
POST /predict
Single Record
{
  "record": {
    "feature1": 12,
    "feature2": 5.6,
    "feature3": 3
  }
}

Multiple Records
{
  "records": [
    { "feature1": 10, "feature2": 4.5, "feature3": 8 },
    { "feature1": 15, "feature2": 6.1, "feature3": 2 }
  ]
}

🛠 Technologies Used

Python 3.10+

FastAPI (high-performance API framework)

Uvicorn (ASGI server)

scikit-learn (ML model)

Pydantic (request validation)

pickle (model loading)

📦 Deployment Ready

This project can be deployed easily on:

Railway

Render

Azure App Service

AWS EC2

Docker Containers

A Dockerfile can be generated on request.

🧾 License

This project is free to use and modify for personal or academic projects.