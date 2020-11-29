from django.shortcuts import render
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

def index(request):
    return render(request, 'webbot/index.htm')


def bot_search(request):
    query = request.GET.get('query')
    book = pd.read_csv("/home/vika/Music/skrip/web/webbot/book_data.csv")    

    filename = '/home/vika/Music/skrip/web/webbot/tfidf'
    output = open(filename,'rb')
    new_dict = pickle.load(output)
    output.close()

    def recommendations(title):
        cosine_similarities = cosine_similarity(new_dict,  new_dict)
        books = book[['book_title', 'image_url']]
        indices = pd.Series(book.index, index = book['book_title']).drop_duplicates()
             
        idx = indices[title]
        sim_scores = list(enumerate(cosine_similarities[idx]))
        sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
        sim_scores = sim_scores[1:6]
        book_indices = [i[0] for i in sim_scores]
        recommend = books.iloc[book_indices]
        for index, row in recommend.iterrows():

            response = requests.get(row['image_url'])
            img = Image.open(BytesIO(response.content))
            plt.figure()
            plt.imshow(img)
            plt.title(row['book_title'])

    hasil = recommendations(query)

    return render(request, 'webbot/index.htm', {'hasil': hasil,'query': query})
