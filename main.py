from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentiment import get_sentiment,get_summary_statistics,analyze_csv_from_filesystem,analyze_single_tweet

import pandas as pd

app = FastAPI()

# CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity; adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PyDantic model for the request body
class TweetRequest(BaseModel):
    text: str
    
@app.post("/analyze_single_tweet")
def analyze_tweet(data:TweetRequest):
    return analyze_single_tweet(data.text)

@app.get("/get_summary")
def dataset_summary():
    try:
        return get_summary_statistics()
    except Exception as e:
        return {"error":str(e)}
    