from keras.models import model_from_json
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
from pydantic import BaseModel
import tensorflow
import random
from fastapi import FastAPI
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import json
# Load the model architecture from the JSON file
with open("chatbot_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
# Load the model weights
loaded_model.load_weights("chatbot_model_weights.h5")

# Compile the loaded model
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nltk.download('punkt')

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

app = FastAPI()

class TextRequest(BaseModel):
    text: str

class KnowledgeRequest(BaseModel):
    question :str
    context :str

@app.post("/")
async def read_root(request_data: TextRequest):
    text = request_data.text     
    print(request_data.text  )
    stemmer = LancasterStemmer()
    with open('intents.json') as file:
        data = json.load(file)
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)
    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)
    training = numpy.array(training)
    output = numpy.array(output)
    inp = request_data.text
    dataToPredict = bag_of_words(inp, words)
    dataToPredict = numpy.array(dataToPredict)
    dataToPredict.shape = (1, len(training[0]))
    results = loaded_model.predict(dataToPredict, batch_size=None)
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    if inp.lower() == "quit":
        return {"response": "bye!"}
    return {"response": random.choice(responses) }

@app.post("/knowledge")
def knowledge_chatbot(request_data: KnowledgeRequest):
    question = request_data.question     
    context = request_data.context
    from transformers import pipeline

    # Load the question-answering model
    qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad", tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad")

    # Define a function to answer questions
    def answer_question(question, context):
        result = qa_pipeline(question=question, context=context)
        return result["answer"]
    # Ask a question
    answer = answer_question(question, context)
    return {
        "question":question,
        "answer": answer
    }