from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json

app = FastAPI()

# Mount the templates and static directories
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

def train_and_predict(train, texts, labels):
    # Convert labels to numerical values
    label_dict = {"jobs related": 0, "casual text related": 1, "questions about me": 2}
    y = np.array([label_dict[label] for label in labels])

    # Vectorize the training text data using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(train)

    # Train a Multinomial Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train_vec, y)

    # Vectorize the new text data
    X_text_vec = vectorizer.transform(texts)

    # Make predictions on the new text data
    y_pred = clf.predict(X_text_vec)

    # Get the unique labels from the training set
    unique_labels = np.unique(y)

    # Print a classification report with precision, recall, and F1-score for the new text data
    ##report = classification_report(y_pred, y, labels=unique_labels, target_names=label_dict.keys())
    ##print("Classification Report (New Text Data):\n", report)

    return y_pred

    # Sample dataset
TRAIN = ["I love my job in data science",
        "This is just a casual conversation.",
        "Can you tell me more about yourself and your interests?",
        "Machine learning and AI are fascinating fields.",
        "I like to write code and I am passionate about life.",
        "what job can I have?",
        "how are you?",
        "can you recommand a job for me","what you know about me?"]

LABELS = ["questions about me", 
            "casual text related",
             "questions about me", 
             "casual text related", 
             "questions about me", 
             "jobs related",
             "casual text related",
             "jobs related","questions about me"]


listQuestions = [
    "What are your favorite hobbies or interests?",
    "Do you enjoy working with people, data, technology, or creative tasks?",
    "What skills do you excel at?",
    "Are you a strong communicator, problem solver, or analytical thinker?",
    "What is your educational background?",
    "Do you have any certifications or degrees that are relevant to your career goals?",
    "What are your short-term and long-term career goals?",
    "Are there specific industries or roles you are interested in pursuing?"
]

def request_intent_model(text):
        # Define the URL you want to send the POST request to
    url = 'http://172.23.0.1:8070/'
    headers = {"Content-Type": "application/json"}
    # Define the data you want to include in the POST request (if any)
    data = {
        "text":text
    }

    # Send the POST request
    response = requests.post(url, data=json.dumps(data), headers=headers)    
    print(response)
    return response.json()

def request_information_store():
    pass

def request_information_model(text, context):
    # Define the URL you want to send the POST request to
    url = 'http://172.23.0.1:8070/knowledge'
    headers = {"Content-Type": "application/json"}
    # Define the data you want to include in the POST request (if any)
    data = {
        "question":text,
        "context": context+"hi"
    }

    # Send the POST request
    response = requests.post(url, data=json.dumps(data), headers=headers)    
    return response.json()

def request_jobs_model(text):
    # Define the URL you want to send the POST request to
    url = 'http://172.23.0.1:8090/'

    # Define the data you want to include in the POST request (if any)
    data = {
        "text":text
    }
    headers = {"Content-Type": "application/json"}
    # Send the POST request
    response = requests.post(url, data=json.dumps(data), headers=headers)    
    print(response)
    return response.json()

class TextRequest(BaseModel):
    text: str
    metadata: list[str]
    context: str

class TextLogin(BaseModel):
    mail: str
    password: str

class TextRegister(BaseModel):
    mail: str
    password: str

@app.post("/chat")
def chat(request_data: TextRequest):
    text = request_data.text   
    context = request_data.context
    meta = request_data.metadata 

    # Call the function to get predicted labels for the new text data
    predicted_labels = train_and_predict(TRAIN, [text], LABELS)

    print("Predicted Labels (New Text Data):", predicted_labels)
    if len(meta) > 0:
        if text == "cancel":
            ## use intent model
            return {"text":"Hi"}
        else:
            ##send to information model and save into database
            return {"text": listQuestions[int(meta[0])]}
    else:
        if predicted_labels == 0:
            return {"text": request_jobs_model(text)}
        elif predicted_labels == 1:
            return {"text": request_intent_model(text)}
        else:
            return {"text": request_information_model(text,context) }

@app.post("/api/login")
async def api_login(request_data: TextLogin):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/api/register")
async def api_register(request_data: TextRegister):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/profile")
async def index(request: Request):
    return templates.TemplateResponse("profile.html", {"request": request})

@app.get("/auth")
async def login(request: Request):
    return templates.TemplateResponse("auth.html", {"request": request})
