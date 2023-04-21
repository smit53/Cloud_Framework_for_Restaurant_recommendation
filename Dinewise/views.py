from django.shortcuts import render
from django.http import HttpResponse
import requests
import nltk

from nltk.stem import PorterStemmer
ps =PorterStemmer()
nltk.download('stopwords')
nltk.download('wordnet')
stopword_list = nltk.corpus.stopwords.words('english')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import pandas as pd

import string

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    return text
def recommend(restuarant,df1,similarity):
    index = df1[df1['name'] == restuarant].index[0]

    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    top_res=[]
    for i in distances[1:10]:
        top_res.append(df1.loc[i[0]][['name','stars', 'review_count', 'categories','address','city','state']])
    return top_res


import math

#CONNECT to data
account_name = 'dinewiseblob'
account_key = '6ePIp7JN5i49LZHTvmgUBSaYP5Q36uXdnjKV6I/I616LnZh/fcfXZTSSGqhXxxX6A30iTRwJ1bom+AStjlDKyg=='
container_name = 'foodserver'

connect_str = 'DefaultEndpointsProtocol=https;AccountName=' + account_name + ';AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'

blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(container_name)

sas_i = generate_blob_sas(account_name = account_name,
                                container_name = container_name,
                                blob_name = 'YELP.csv',
                                account_key=account_key,
                                permission=BlobSasPermissions(read=True),
                                expiry=datetime.utcnow() + timedelta(hours=1))
    
sas_url = 'https://' + account_name+'.blob.core.windows.net/' + container_name + '/' + 'YELP.csv' + '?' + sas_i

df = pd.read_csv(sas_url)

#df = pd.read_csv('/Users/aarushidua/Documents/ECC/Dinewise_E516/Dinewise/foodserver/Dinewise/YELP.csv')

# Create a geocoder instance

def search_restaurants(lat, lng, df):
    df = df.drop_duplicates(subset='business_id')
    # Calculate the distance between each restaurant and the user's location
    df['distance'] = df.apply(lambda x: distance(lat, lng, x['latitude'], x['longitude']), axis=1)
    
    # Sort the restaurants by star rating and review count to obtain the top 10000 restaurants
    df = df.sort_values(by=['distance','stars', 'review_count'], ascending=[True,False, False]).iloc[:10000]
    
    # Return the list of restaurants, along with their star ratings, review counts, and distance from the user's location
    return df[['name', 'stars', 'review_count', 'categories', 'text','address','city','state']]

def distance(lat1, lng1, lat2, lng2):
    # Calculate the distance between two locations using the Haversine formula
    R = 6371  # radius of the earth in km
    dLat = math.radians(lat2 - lat1)
    dLng = math.radians(lng2 - lng1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLng / 2) * math.sin(dLng / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c  # distance in km
    return d


def home(request):
    return render(request, 'Dinewise/recommendation.html')

def location(request):
    address = request.GET['address']
    # Django view code to retrieve selected values
    food_types = request.GET.getlist('food-type')
    choice = '|'.join(food_types)
    print("choice")



    #location = geolocator.geocode(address
    api_key = "f761c1a2d4544e3481713f9d93e83e88"
    url = f"https://api.opencagedata.com/geocode/v1/json?q={address}&key={api_key}"

    response = requests.get(url).json()
    location = response["results"][0]["geometry"]
    
    latitude=location["lat"]
    longitude=location["lng"]
    #print(df.head())
    nearest_restaurants = search_restaurants(latitude, longitude, df)
    #print(nearest_restaurants.head())
    nearest_restaurants['tags'] = nearest_restaurants['name']+ nearest_restaurants['categories']+nearest_restaurants['text']
    
    nearest_restaurants['tags']=nearest_restaurants['tags'].apply(remove_punctuations)
    nearest_restaurants['tags']=nearest_restaurants['tags'].apply(lambda x: x.lower())

    nearest_restaurants['tags']=nearest_restaurants['tags'].apply(lambda x: ' '.join([i for i in x.split() if i not in stopword_list]))
    nearest_restaurants['tags']=nearest_restaurants['tags'].apply(lambda x:' '.join([ps.stem(i) for i in x.split()]))
    
    nearest_restaurants=nearest_restaurants.reset_index()
    tfidf = TfidfVectorizer().fit_transform(nearest_restaurants['tags']).toarray()
    similarity = cosine_similarity(tfidf)
    res_name = nearest_restaurants[nearest_restaurants["categories"].str.contains(choice, case=False)].iloc[0]
    print(res_name)
    print("--------------------------")
    top_res=recommend(res_name['name'],nearest_restaurants,similarity)
    #print(top_res)
    # convert the DataFrame to a list of dictionaries
    #data = nearest_restaurants.to_dict("records")    

    # pass the data to the template using the context dictionary
    context = {"top_res": top_res}
    # Pass the nearest_restaurants data to a template for rendering
    return render(request, 'Dinewise/results.html',  context)
