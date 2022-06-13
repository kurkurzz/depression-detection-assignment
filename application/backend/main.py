import re
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
nlp = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])

def clean_text(text):
    text = text.lower()
    # remove @username, #hashtag, url
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", text).split())
    doc = nlp(text)

    # lemmatization, remove stop words and other unrelevant character
    text = ' '.join(token.lemma_ for token in doc if
                not token.is_punct
                and not token.is_currency
                and not token.is_digit
                and not token.is_space
                and not token.is_stop
                and not token.like_num
                and not token.pos_ == "PROPN"
                    )
    return text

@app.get('/', status_code=status.HTTP_200_OK)
async def ping(request: Request):
	return {
		'status': 'success',
		'made by': 'hafiz <3'
	}

@app.get('/predict', status_code=status.HTTP_200_OK)
async def predict(request: Request, text: str = None):
	if text:
		text = clean_text(text)
		prediction = model.predict([text])
			
		return {
				'text': text,
				'prediction': prediction[0]
			}