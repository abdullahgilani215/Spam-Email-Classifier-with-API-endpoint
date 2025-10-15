import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import warnings
from sklearn.exceptions import InconsistentVersionWarning

app = FastAPI()

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)



model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class Email(BaseModel):
    text:str
@app.get("/")
def index():
    return{"message":"Spam Email Classifier API is running"}

@app.post("/predict")
def predict(email:Email):
    text = email.text
    email_features = vectorizer.transform([text])
    prediction = model.predict(email_features)[0]
    label = "spam" if prediction == 1 else "not spam"
    return {"prediction":label}
