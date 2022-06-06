from fastapi import FastAPI, status, Request
from fastapi.middleware.cors import CORSMiddleware
import spacy
import joblib


app = FastAPI()

origins = [
    "https://twitter.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("rfc.joblib")

@app.get('/', status_code=status.HTTP_200_OK)
async def ping(request: Request):
	return {
		'status': 'success',
		'made by': 'hafiz <3'
	}

@app.get('/predict', status_code=status.HTTP_200_OK)
async def predict(request: Request, text: str = None):
	if text:
		prediction = model.predict([text])
			
		return {
				'text': text,
				'prediction': prediction[0]
			}