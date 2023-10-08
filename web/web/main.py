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

class TextReddit(BaseModel):
    jobs: str

# Define the URL and headers
def reddit(paramter):
  headers = {
    "User-Agent": "ChangeMeClient/0.1 by YourUsername",
    "Cookie":"csv=2; edgebucket=LZMx5O2w9aXIB4hmyx; csrf_token=d3f56f539b60608ba51c0741e01948be; token_v2=eyJhbGciOiJSUzI1NiIsImtpZCI6IlNIQTI1NjpzS3dsMnlsV0VtMjVmcXhwTU40cWY4MXE2OWFFdWFyMnpLMUdhVGxjdWNZIiwidHlwIjoiSldUIn0.eyJzdWIiOiJsb2lkIiwiZXhwIjoxNjk2Nzc4Njk3Ljk1MDY3NywiaWF0IjoxNjk2NjkyMjk3Ljk1MDY3NywianRpIjoiR2I3SE5HVExmelo1REpoY2RWTy14bzcxMXBYOWtRIiwiY2lkIjoiMFItV0FNaHVvby1NeVEiLCJsaWQiOiJ0Ml9sYTFydnB6cTkiLCJsY2EiOjE2OTY2ODg2OTIyOTYsInNjcCI6ImVKeGtrZEdPdERBSWhkLWwxejdCX3lwX05odHNjWWFzTFFhb2szbjdEVm9jazcwN2NENHBIUDhuS0lxRkxFMnVCS0c0eXBsNzgxNFdMSVZNMDVRR3RheEFrcWIwSkRXV2Q1b1NGV3hHNW5LbEhTczBlR0NhVXVXU3VTMzB1TFFKemQxWTlPekVxTXBsNVVGVm9QVlVqVzFNWVh0aWZMT3gycENLNjNJcUUxZ1d5bWZ4b2g5eTlkWS1mN2JmaEhZd3JLZ0tEX1RPdUZWd1lfSERGSFpfVDAxNnRpNVkxTjdyUVdxZjYzRzc5bG16ME96Y2ZxN25yNDFrWEk2aC1RbjI3NjVmUWdjZWVWNW0xQUdNV01PUE11eklPdnlyRHFCeVFRRmpDZUxUQ09USzVkdF9DYlpyMDdfRzdSTV9mRFBpcGpmODFnelVjd25pMEtmeDlSc0FBUF9fSV9yYXVRIiwiZmxvIjoxfQ.gEq_xNYv7iH9aL8c1VOHjYCyNUeRU-fvpR83K3Fo0DCTyEOMggaf-T6qNqbujWxbBSihbKKFW5QSs1_VZQ55j4B3LjsCHUcmGDQFGaWJmZ06uiaVjbFEIABFILfGI0L09SiovqhTnjLUjoNBB3z8xUgO8zpWz4PYiM4ojh04_oqaoYP4nk_QYPjuABZbuNAkScxIvL9KpRD2ctHzhkkYZF41dDSg62i9DTFlgC3xXsoJuxQapNJ-qHw29h5ZoGJbpMsydeXam8hmHgPqvTdSjWHqkmGon5MOV4FMJLYpay2jx1EhY-xDPegvhdDyNpel3jX-yG4xlnSa-PngLhnYGw; loid=000000000la4yfh1d8.2.1696691761491.Z0FBQUFBQmxJWG9JdHY3N1Z3QTZEMndOVTdKMzZCVm5LWHJkd3hsQjkzc1Z1N3RNWFhhX2lRYUZCV01tR3B5VFA3VUVpNWU4RFgyRWhMU2xZX0Fyd1ViVklXVHl3a1lQemt1WWV5UG1VcXJ3Z1YxTXlzaElIYVV4S3pJNVB0VnZiWFhRbHJscUVrNDk; session_tracker=bbfajmcbhaqaciaqff.0.1696699700473.Z0FBQUFBQmxJWlUwUWI1NWN1eDJFMlpjTEtKcDR5Wmlxc1BYeXhDWl83YVJfbzV0NGNQdXRGVHNzRXpGRUV1YnNDYU42LThlRjFpb1Y1d1ZFeFhILWR0MjZta0ZzbzEwWjBuN3NDMUk2a0VSRnowQ0Q3WUhmVURGV3VBLUNvRERnNW9RQzg5eE11QXA",
    "Cache-Control":"no-cache",
    "Accept":"*/*",
    "Accept-Encoding":"gzip, deflate, br",
    "Connection":"keep-alive",
    "Authorization": "bearer eyJhbGciOiJSUzI1NiIsImtpZCI6IlNIQTI1NjpzS3dsMnlsV0VtMjVmcXhwTU40cWY4MXE2OWFFdWFyMnpLMUdhVGxjdWNZIiwidHlwIjoiSldUIn0.eyJzdWIiOiJ1c2VyIiwiZXhwIjoxNjk2Nzc4NzkyLjMzNzg3NCwiaWF0IjoxNjk2NjkyMzkyLjMzNzg3NCwianRpIjoiX2xWR2Z0enpWRmZuaENEajFad0Roc2NSeTFGcEdnIiwiY2lkIjoianJTZ0l4bWpwVGlKak1qU0tPRVJ2QSIsImxpZCI6InQyX2xhNHlmaDFkOCIsImFpZCI6InQyX2xhNHlmaDFkOCIsImxjYSI6MTY5NjY5MTc2MTQ5MSwic2NwIjoiZUp5S1Z0SlNpZ1VFQUFEX193TnpBU2MiLCJmbG8iOjl9.MPzPdcn89S9nohb4Ea1VOK2Kn04nhynDMOaNvKSdwdZddJeWbOGItNGwJxmfrCOSANUWf1H_NfyETh2Rsh9YjmWYGUJ4RZE8Li6olhmkcCgUFvrbgwbgePvNSVhgybyJdEoldEANHsig8oqvU_wifUecQ4IAoeAbdOpZtplLHVzRBATx5K0Kk7fzS-prE-5VNitLMRFv5f3RH-32kJEj_rWGj6hGnEMim0xJ5YSCGNbPdA5KCLdXWY_YDzawlmWU_WSuWQXeE_mcFsO-E9TqTGN0PZgc9FNrnyQTldbkzNBnFMpJ5wIjxvm6xa5xzsrs8UCMQ0wpIQy3Y37rusL7ng",
  }

  url = "https://oauth.reddit.com/subreddits/search?q="+paramter
  response = requests.get(url, headers=headers)
  data = response.json()
  resp=[]
  # Print the response content (you can also parse it as needed)
  print(data.keys())  # Print the keys in the response dictionary
  children = data["data"]["children"]
  for child in children:
    subreddit_data = child["data"]
    resp.append({
        "name":subreddit_data["display_name"],
        "title": subreddit_data["title"],
        "subscribers": subreddit_data["subscribers"]
    })

  return resp

@app.post("/api/reddit")
def api_reddit(request_data: TextReddit):
    jobname = request_data.text  
    print(jobname) 
    return reddit(jobname)

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
            return {"text": request_jobs_model(context)}
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
