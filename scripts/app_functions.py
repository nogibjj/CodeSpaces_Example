import os 

import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.stem import WordNetLemmatizer

import torch
import numpy as np
# nltk.download('omw-1.4')

def tokenize(sentence,method='nltk'):
# Tokenize and lemmatize text, remove stopwords and punctuation

    punctuations = string.punctuation
    stopwords = list({"really", "sometimes", "go", "since", "whither", "they", "its", "them", "well", "meanwhile", "seems", "and", "latterly", "regarding", "somehow", "sixty", "whole", "anyway", "else", "few", "m", "beside", "to", "namely", "someone", "see", "moreover", "wherein", "for", "former", "bottom", "it", "next", "six", "along", "once", "might", "whenever", "below", "another", "yourself", "each", "just", "ourselves", "everyone", "any", "across", "get", "that", "eight", "we", "which", "therefore", "may", "s", "keep", "among", "give", "such", "are", "indeed", "everywhere", "same", "herself", "yourselves", "alone", "were", "was", "take", "seem", "say", "why", "show", "between", "during", "elsewhere", "or", "though", "forty", "made", "used", "others", "whereafter", "formerly", "several", "via", "does", "please", "three", "also", "fifty", "afterwards", "s", "noone", "do", "perhaps", "further", "i", "beforehand", "myself", "empty", "ll", "yet", "thereby", "been", "both", "never", "put", "without", "him", "a", "nothing", "thereafter", "make", "then", "whom", "must", "sometime", "against", "through", "being", "four", "back", "become", "our", "himself", "because", "anything", "re", "nor", "therein", "due", "until", "own", "ca", "most", "now", "while", "of", "only", "am", "itself", "too", "m", "nobody", "if", "one", "whereas", "twelve", "together", "can", "who", "even", "be", "she", "besides", "herein", "off", "d", "last", "no", "whereupon", "the", "m", "out", "hereupon", "by", "us", "already", "became", "here", "hers", "onto", "beyond", "down", "enough", "did", "some", "over", "serious", "quite", "move", "around", "nowhere", "amongst", "but", "so", "wherever", "twenty", "often", "part", "again", "where", "re", "within", "at", "nt", "yours", "front", "unless", "could", "anyone", "third", "whatever", "doing", "d", "nevertheless", "before", "rather", "fifteen", "her", "me", "thereupon", "mostly", "throughout", "hence", "re", "mine", "ten", "hundred", "nine", "call", "when", "about", "will", "whereby", "this", "upon", "you", "should", "always", "themselves", "not", "has", "behind", "on", "anywhere", "side", "their", "hereby", "latter", "after", "ve", "none", "these", "name", "nt", "every", "although", "s", "however", "he", "becoming", "how", "whose", "still", "hereafter", "whether", "towards", "more", "everything", "whoever", "seemed", "cannot", "up", "otherwise", "in", "would", "under", "done", "thence", "whence", "seeming", "either", "other", "with", "into", "amount", "five", "much", "re", "except", "his", "thus", "ll", "what", "almost", "becomes", "least", "ever", "above", "is", "first", "there", "somewhere", "top", "ve", "ve", "than", "nt", "have", "toward", "per", "all", "ours", "full", "d", "anyhow", "as", "ll", "many", "various", "your", "had", "eleven", "from", "something", "less", "those", "using", "an", "two", "my", "very", "neither"})

    if method=='nltk':
        # Tokenize
        tokens = nltk.word_tokenize(sentence,preserve_line=True)
        # Remove stopwords and punctuation
        tokens = [word for word in tokens if word not in stopwords and word not in punctuations]
        # Lemmatize
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
        tokens = " ".join([i for i in tokens])
    else:
        pass
    return tokens

def predict_rating(model, userId, productId, sentiment, device):
    # Encode genre
    userId = np.array(userId).reshape(-1)
    productId = np.array(productId).reshape(-1)
    sentiment = np.array(sentiment).reshape(-1)

    # Get predicted rating
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        X = torch.Tensor([userId,productId, sentiment]).long().view(1,-1)
        X = X.to(device)
        pred = model.forward(X)
        return pred

def generate_recommendations(remove_products, prod_sent_dict, products, product_encoder, model,userId, device):
    # Get predicted ratings for every movie
    pred_ratings = {}
    for product in products:
        cur_product = product_encoder.transform([product])[0]
        try:
            sentiment = prod_sent_dict[product]
        except:
            sentiment = np.random.randint(0, 2)
        pred = predict_rating(model,userId, cur_product, sentiment, device)
        pred_ratings[product] = pred.detach().cpu().item()

    # Sort movies by predicted rating
    recs_sort = dict(sorted(pred_ratings.items(), reverse = True, key=lambda item: item[1]))
    # print(recs_sort)
    for prod in remove_products:
        recs_sort.pop(prod)
    recs = list(recs_sort.keys())
    top5_recs = recs[:5]

    return top5_recs