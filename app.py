import pandas as pd
from flask import Flask, request
from flask_cors import CORS, cross_origin
import pickle
import json
import numpy as np

# load model
indices = pickle.load(open('book_indices_new.pkl','rb'))
splitted_arrs_0 = pickle.load(open('book_similarity_new1.pkl','rb'))
splitted_arrs_1 = pickle.load(open('book_similarity_new2.pkl','rb'))
splitted_arrs_2 = pickle.load(open('book_similarity_new3.pkl','rb'))
book_img = pickle.load(open('book_other_details_new.pkl','rb'))
titles = pickle.load(open('book_title_array_new.pkl','rb'))

cosine_sim_corpus = np.concatenate((splitted_arrs_0, splitted_arrs_1, splitted_arrs_2))

# app
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# routes
@app.route('/', methods=['GET','POST'])
@cross_origin()

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
    authors = []
    publication_year = []
    for item in recommended_books:
        images.append(book_img["image_url"].loc[item])
        authors.append(book_img["authors"].loc[item])
        publication_year.append(book_img["original_publication_year"].loc[item])

    result={'title':recommended_books,
       'images':images,
       'authors': authors,
       'publication_year': publication_year
       }

    resultjson = json.dumps(result)



    #return result

    # send back to browser

    return resultjson

if __name__ == '__main__':
    app.run()