from fastapi import FastAPI
from models import sa_pipeline, query_index
from pydantic import BaseModel

class QARequest(BaseModel):
    question: str

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    score: float

app = FastAPI()

@app.post('/question-answering')
def query(query: QARequest):
    data = query.dict()
    return query_index(data['question'])

@app.post('/sentiment-analysis', response_model=SentimentResponse)
def query(query: SentimentRequest):
    data = query.dict()
    return sa_pipeline(data['text'])[0]