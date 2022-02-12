import pandas as pd
from flask import Flask, request
import pickle
import json

# load model
indices = pickle.load(open('book_indices.pkl','rb'))
cosine_sim_corpus = pickle.load(open('book_similarity.pkl','rb'))
book_img = pickle.load(open('book_images.pkl','rb'))
titles = pickle.load(open('book_title_array.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['GET','POST'])

def predict():
    # get data
    name = request.json['name']

    recommended_books = []
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim_corpus[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    for i in book_indices:
        recommended_books.append(titles[i])
    images = []
    for item in recommended_books:
        images.append(book_img["image_url"].loc[item])

    result={'title':recommended_books,
       'images':images}

    resultjson = json.dumps(result)



    #return result

    # send back to browser

    return resultjson

if __name__ == '__main__':
    app.run()