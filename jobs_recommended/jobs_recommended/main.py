import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib  # for saving the model
from fastapi import FastAPI
from fastapi import BackgroundTasks
from pydantic import BaseModel

vectorizer = None
knn = None
jobs_df = None

def read_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df[['Job Title', 'Job Description']]
# Load data and extract job descriptions
jobs_df = read_data("jobs.csv")
job_descriptions = jobs_df['Job Description']
# Vectorize job descriptions using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
job_descriptions_vectorized = vectorizer.fit_transform(job_descriptions)
# Initialize KNN model
knn = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
knn.fit(job_descriptions_vectorized)


app = FastAPI()

class TextRequest(BaseModel):
    text: str



# Define a function to run on startup
async def on_startup():
    return {"knn":knn }

@app.on_event("startup")
async def startup_event():
    await on_startup()

@app.post("/")
async def read_root(request_data: TextRequest):
    text = request_data.text
    # Vectorize the example text
    text_vectorized = vectorizer.transform([text])
    # Find the nearest neighbors (similar job descriptions)
    similar_jobs_indices = knn.kneighbors(text_vectorized, n_neighbors=13, return_distance=False)
    # Get the job titles and descriptions of the similar jobs
    similar_jobs = jobs_df.iloc[similar_jobs_indices[0]]

    
    return {"jobs": similar_jobs.drop_duplicates(subset="Job Title") }


if __name__ == "__main__":
    import uvicorn

    # Start the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8090)
