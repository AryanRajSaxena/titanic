from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

app = FastAPI()

class PredictionInput(BaseModel):

    Pclass: int
    Age: int
    co_travelers : int

model_path = R"C:\Users\hp\OneDrive\Desktop\machine learning\titanic\titanic-mlops\model\model.joblib"
model = load(model_path)

@app.get("/")
def home():
    return "Working good !"

@app.post("/predict")
def predict(input_data : PredictionInput):
    features = [
        input_data.Pclass,
        input_data.Age,
        input_data.co_travelers
    ]
    prediction = model.predict([features])[0].item()
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
