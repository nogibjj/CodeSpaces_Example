import torch
import streamlit as st
import pandas as pd
from scripts.app_functions import *
import pickle
from hybrid_model import NNHybridFiltering
from dblib.querydb import querydb

# load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# title of streamlit app
st.title('Amazon Fine Food Recommendations')

# load data
df = pd.read_csv('data/processed/app_data_test.csv')

# load all unique values of products and ids
productIds = set(df.ProductId)
userIDs = set(df.UserId)

# ask user to input fields needed for predictions
user_id = st.selectbox('Select User Profile:', userIDs)
# product_id = st.selectbox('Select Product ID:', productIds)
# score = st.select_slider('Enter Review Score:',
#      options=[1,2,3,4,5])
# review = st.text_input('Enter Review:')

# if submitted, make predictions
if st.button('Sign In'):
    st.text(f'Top items purchased by {user_id}:')
    sort_prod = pd.DataFrame(querydb(f"SELECT * FROM app_data_test_csv WHERE UserId = '{user_id}' ORDER BY Score DESC"))
    sort_prod.columns = ["ProductId", "UserId", "sentiment", "Score"]
    sort_prod.loc[:,["sentiment", "Score"]] = sort_prod.loc[:,["sentiment", "Score"]].astype(float)
    #sort_prod = df[df['UserId']==user_id].sort_values(by='Score', ascending=False)
    prod_sent_dict = dict(zip(sort_prod.ProductId, sort_prod.sentiment))
    usr_prod = list(sort_prod['ProductId'])[:3]
    top_items = ''
    for item in usr_prod:
        top_items += ' amazon.com/dp/'
        top_items += item
        top_items += '\n'
    st.text(f'{top_items}')
    # review_tokenize = tokenize(review).split()

    # tfidf = pickle.load(open('models/tfidf.pickle', 'rb'))
    # review_tfidf = tfidf.transform(review_tokenize)

    # logreg = pickle.load(open('models/tfidf_reg.pickle', 'rb'))
    # sentiment = logreg.predict(review_tfidf)[0]+1

    # load model
    model = torch.load('models/model.pt', map_location=torch.device('cpu'))
    user_encoder = pickle.load(open('models/user_encoder.pickle', 'rb'))
    product_encoder = pickle.load(open('models/product_encoder.pickle', 'rb'))

    user_id = user_encoder.transform([user_id])[0]
    recs = generate_recommendations(
        usr_prod,
        prod_sent_dict,
        list(df.ProductId.unique()),
        product_encoder,
        model,
        user_id,
        device)

    st.text('')
    st.text('We think you would also like:')
    string_recs = ''
    for rec in recs:
        string_recs += ' amazon.com/dp/'
        string_recs += rec
        string_recs += '\n'
    st.text(f'{string_recs}')
