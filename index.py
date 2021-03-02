# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 13:48:16 2021

@author: eliot
"""

import base64
import datetime
import io
import pickle
import re
import numpy as np
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objects as go
import plotly.express as px

from app import app, server 


import pandas as pd


import spacy
import pt_core_news_sm
nlp = pt_core_news_sm.load()

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import TruncatedSVD
from spacy.lang.pt.stop_words import STOP_WORDS as STOP_WORDS_Cluster
from spacy.lang.pt.stop_words import STOP_WORDS as STOP_WORDS_Sentiment


#Loading the models : 
filename_clustering = f'model/model_cluster_2.pkl'
filename_sentiment = f'model/model_sentiment_2.pkl'
model_clustering = pickle.load(open(filename_clustering, 'rb'))
model_sentiment = pickle.load(open(filename_sentiment, 'rb'))


        #Removing stop words and lemmatize the reviews:
STOP_WORDS_Cluster.add("e")
STOP_WORDS_Cluster.add("o")
STOP_WORDS_Cluster.add("produto")
STOP_WORDS_Cluster.add("prazo")
STOP_WORDS_Cluster.add("lannister") 
STOP_WORDS_Cluster.add("nao")
STOP_WORDS_Cluster.add("perfazer")
STOP_WORDS_Cluster.add("pra")
STOP_WORDS_Cluster.add("ok")
STOP_WORDS_Cluster.add("outro")
STOP_WORDS_Cluster.add("errar")
STOP_WORDS_Cluster.add("haver")
STOP_WORDS_Cluster.add("dever")
STOP_WORDS_Cluster.add("entrar")
STOP_WORDS_Cluster.add("conseguir")
STOP_WORDS_Cluster.add("correar")
STOP_WORDS_Cluster.add("fiscal")
STOP_WORDS_Cluster.add("atar")
STOP_WORDS_Cluster.add("vir")
STOP_WORDS_Cluster.add("entregar")
STOP_WORDS_Cluster.add("chegar")
STOP_WORDS_Cluster.add("comprar")
STOP_WORDS_Cluster.add("number")
STOP_WORDS_Cluster.add("recomendar")


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


    # Fonctions pour le pipeline : 
def re_special_chars(text_list):
    return [re.sub('\W', ' ', r) for r in text_list]
def re_breakline(text_list):
    return [re.sub('[\n\r]', ' ', r) for r in text_list]
def re_hiperlinks(text_list):
    pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return [re.sub(pattern, ' link ', r) for r in text_list]
def re_dates(text_list):
    pattern = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
    return [re.sub(pattern, ' date ', r) for r in text_list]
def re_money(text_list):
    pattern = '[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
    return [re.sub(pattern, ' money ', r) for r in text_list]
def re_numbers(text_list):
    return [re.sub('[0-9]+', ' number ', r) for r in text_list]
def re_whitespaces(text_list):
    white_spaces = [re.sub('\s+', ' ', r) for r in text_list]
    white_spaces_end = [re.sub('[ \t]+$', '', r) for r in white_spaces]
    return white_spaces_end
def re_negation(text_list):
    return [re.sub('([nN][ãÃaA][oO]|[ñÑ]| [nN] )', ' negação ', r) for r in text_list]

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False  #Can be true 
    ),
    html.Div(
        children=[
            html.Div(
                children=[
                    html.P(children="Select your Product category:"),
                    dcc.Dropdown(
                        id="select-product-name"),
                    dcc.RadioItems(
                            id="filter_topic_selector",
                            options=[      
                                {"label": "All ", "value": "all"},
                                {"label": "Livraison", "value": "Livraison"},
                                {"label": "Qualité ", "value": "Qualité"},
                                {"label": "Autre ", "value": "Autre"},
                            ],
                            value="all",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                                   ),
                    html.Div(
                        id="product-name-div",
                        # You can add some inline styling like this. This style
                        # will be applied only to this div.
                        style={"margin-top": "18px"}
                    ),
                ],
                className="col-md-4"
            ),
                ],
        className="row"
    ),
    
    
    html.Div(children=[
                    html.H3(
                        children="General Sentiment :",
                        style={"text-align": "center"}
                    ),
                    #dcc.Graph(id="happy-stats"),
                    #dcc.Graph(id="sad-stats"),
                    dcc.Graph(id="product-stats"),
                ],
                className="col-md-8"
            ),
        
        html.Div([
        html.Div([
            html.H3('20% MOST happier customers talk about :'),
            dcc.Graph(id='happy-stats')
            ], className="six columns"),

        html.Div([
            html.H3('20% LESS happier customers talk about :'),
            dcc.Graph(id='sad-stats')
        ], className="six columns"),
    ], className="row"),
    
    
            html.Div([
        html.Div([
            html.H3('Top 10 product categories:'),
            dcc.Graph(id='top5-stats')
            ], className="six columns"),

        html.Div([
            html.H3('Flop 10 product categories :'),
            dcc.Graph(id='flop5-stats')
        ], className="six columns"),
    ], className="row"),


], className="container-md")


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        else:
            raise('FileNotCSV')
    except Exception as e:
        print(e)
        return html.Div([
           'There was an error processing this file.'
        ])

    return df

            

@app.callback(Output('select-product-name', 'options'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        df = parse_contents(list_of_contents, list_of_names, list_of_dates)
        return [{"label": str(category), "value": str(category)} for category in df['product_category_name'].unique()]
    else:
        raise('FileIsEmpty')
    
    
@app.callback(Output('product-stats','figure'),
              Output('happy-stats','figure'),
              Output('sad-stats','figure'),
              Output('top5-stats','figure'),
              Output('flop5-stats','figure'),
              Input('select-product-name', 'value'),
              Input('filter_topic_selector', 'value'),
              Input('upload-data','contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified')
             )
def update_graph(product_cat, topic, list_of_contents, list_of_names, list_of_dates): 
    if list_of_contents is not None:
        df = parse_contents(list_of_contents, list_of_names, list_of_dates)
        
                    # Créer deux pipeline + appliquer 2 modèles + fusionner 
    
    

    
    
    # Pipeline 1 : Clusterisation :
        #applying all the preprocess functions:
    df_clustered = df.copy()
    df_clustered = df_clustered.dropna()
    df_clustered['review_comment_message'] = re_special_chars(df_clustered['review_comment_message'])
    df_clustered['review_comment_message'] = re_breakline(df_clustered['review_comment_message'])
    df_clustered['review_comment_message'] = re_hiperlinks(df_clustered['review_comment_message'])
    df_clustered['review_comment_message'] = re_dates(df_clustered['review_comment_message'])
    df_clustered['review_comment_message'] = re_money(df_clustered['review_comment_message'])
    df_clustered['review_comment_message'] = re_numbers(df_clustered['review_comment_message'])
    df_clustered['review_comment_message'] = re_whitespaces(df_clustered['review_comment_message'])
    df_clustered['review_comment_message'] = [item.lower() for item in df_clustered['review_comment_message']]
    
        #tokenization : 
    def list(sentence):
        words = []
        for word in sentence:
            words.append(words)
        return words
            

    df_clustered['clean_tokens'] = df_clustered['review_comment_message'].apply(lambda x: nlp(x))
    

    df_clustered['clean_tokens'] = df_clustered['clean_tokens'].apply(lambda x: [token.lemma_ for token in x if token.text not in STOP_WORDS_Cluster and token.lemma_ not in STOP_WORDS_Cluster])
    df_clustered["clean_reviews"] = [" ".join(x) for x in df_clustered['clean_tokens']]


        # TF-IDF :
    cv = TfidfVectorizer(stop_words=STOP_WORDS_Cluster)
    dtm = cv.fit_transform(df_clustered["clean_reviews"])
    dtm = dtm.toarray()
    X_df = pd.DataFrame(dtm, 
             columns=cv.get_feature_names(), 
             index=["item_{}".format(x) for x in range(df_clustered.shape[0])] )
    

        # Application du modèle :
    lsa = model_clustering.fit_transform(dtm)  #sinon dtm 
    topic_encoded_df = pd.DataFrame(lsa, columns = ["topic_" + str(i) for i in range(lsa.shape[1])])
    topic_encoded_df["documents"] = df_clustered['clean_reviews']
    def extract_main_topics(x):
        topics = np.abs(x)
        main_topic = topics.sort_values(ascending=False).index[0]
        return main_topic

    topic_encoded_df.loc[:, 'main_topic'] = np.nan

    for i, row in topic_encoded_df.iloc[:,:-2].iterrows():
        topic_encoded_df.loc[i, 'main_topic'] = extract_main_topics(row)
    df_clustered.reset_index(inplace=True)
    df_clustered.drop('index', axis=1, inplace=True)
    df_clustered['topic'] = topic_encoded_df['main_topic']
    cols = ['id', 'review_comment_message', 'topic', 'product_category_name']
    df_clustered = df_clustered[cols]
    
    
    
    
    #Pipeline 2 : Sentimentalisation :
    df_sentiment = df.copy()
    df_sentiment = df_sentiment.dropna()
    
        #applying all the preprocess functions:
    df_sentiment['review_comment_message'] = re_special_chars(df_sentiment['review_comment_message'])
    df_sentiment['review_comment_message'] = re_breakline(df_sentiment['review_comment_message'])
    df_sentiment['review_comment_message'] = re_hiperlinks(df_sentiment['review_comment_message'])
    df_sentiment['review_comment_message'] = re_dates(df_sentiment['review_comment_message'])
    df_sentiment['review_comment_message'] = re_money(df_sentiment['review_comment_message'])
    df_sentiment['review_comment_message'] = re_numbers(df_sentiment['review_comment_message'])
    df_sentiment['review_comment_message'] = re_whitespaces(df_sentiment['review_comment_message'])
    df_sentiment['review_comment_message'] = re_negation(df_sentiment['review_comment_message'])
    df_sentiment['review_comment_message'] = [item.lower() for item in df_sentiment['review_comment_message']]
    
        #Tokenization
    df_sentiment['clean_reviews'] = df_sentiment['review_comment_message'].apply(lambda x: nlp(x))
    
        #Removing Stop Words and Lemmatize
    STOP_WORDS_Sentiment.add("e")
    STOP_WORDS_Sentiment.add("o")
    df_sentiment['clean_reviews'] = df_sentiment['clean_reviews'].apply(lambda x: [token.lemma_ for token in x if token.text not in STOP_WORDS_Sentiment and token.lemma_ not in STOP_WORDS_Sentiment])
    
    # Put back tokens into one single string
    df_sentiment["clean_reviews_encoded"] = [" ".join(x) for x in df_sentiment['clean_reviews']]
    
    # TF-IDF vector
    cv = TfidfVectorizer(max_features=300)
    dtm = cv.fit_transform(df_sentiment["clean_reviews_encoded"])
    dtm = dtm.toarray()

    # Create a dataframe with tf-idf to see how it looks like:
    X_df = pd.DataFrame(dtm, 
             columns=cv.get_feature_names(), 
             index=["item_{}".format(x) for x in range(df_sentiment.shape[0])] )
    X_df.reset_index(inplace=True)
    X_df.drop('index', axis=1, inplace=True)
    
    #Making the prediction
    X_df['sentiment'] = X_df.apply(lambda s: model_sentiment.predict_proba(s.values[None])[0][1], axis=1)  
    cols = ['id', 'review_comment_message', 'product_category_name']
    df_sentiment = df_sentiment[cols]
    df_sentiment.reset_index(inplace=True)
    df_sentiment.drop('index',axis=1, inplace=True)
    df_sentiment['sentiment'] = X_df['sentiment']
    
    #Fusion des deux df :
    df = pd.merge(df_clustered, df_sentiment[['id', 'sentiment']], on='id', how='left')
    
    #A remettre au tout début de la fonction si on veut éviter le preprocess : 
    df_count = df.groupby('product_category_name').count()
    df_count.reset_index(inplace=True)
    df_count.rename(columns={'id':'count'}, inplace=True)
    df = pd.merge(df,df_count[['product_category_name', 'count']], on='product_category_name', how='left')
    df.sort_values(by='sentiment', ascending=False, inplace=True)
    df.topic = df.topic.apply(lambda x : 'Livraison' if x == 'topic_0' else 'Qualité' if x == 'topic_1' else "Autre" )
    df_20_happy = df.iloc[:int(len(df)*0.2),:]
    df_20_unhappy = df.iloc[int(len(df)*0.8):,:]
    df_group_cat_topic = df.groupby(['product_category_name', 'topic']).mean()
    df_group_cat_topic.reset_index(inplace=True)
    df_group_cat_topic_TOP = df_group_cat_topic.groupby('product_category_name').mean().sort_values(by='count', ascending=False)
    df_group_cat_topic_FLOP = df_group_cat_topic.groupby('product_category_name').mean().sort_values(by='count')

    
    if product_cat is not None:
        df = df[df['product_category_name'] == product_cat]
        df_20_happy = df_20_happy[df_20_happy['product_category_name']== product_cat]
        df_20_unhappy = df_20_unhappy[df_20_unhappy['product_category_name']== product_cat]

    if topic != 'all':
        df = df[df['topic']== topic] 
        df_20_happy = df_20_happy[df_20_happy['topic']== topic]
        df_20_unhappy = df_20_unhappy[df_20_unhappy['topic']== topic]


        
        
    fig = px.box(df, x=df.product_category_name, y=df.sentiment)
    fig_1 = px.pie(df_20_happy, values='sentiment', names='topic', color='topic', color_discrete_map={'Livraison':'blue',
                                 'Qualité':'red',
                                 'Autre':'green'})
    fig_2 = px.pie(df_20_unhappy, values='sentiment', names='topic', color='topic', color_discrete_map={'Livraison':'blue',
                                 'Qualité':'red',
                                 'Autre':'green'})
    
    fig_3 = px.bar(df_group_cat_topic_TOP.iloc[:10,:], x=df_group_cat_topic_TOP.index[:10], y='count')
    
    fig_4 = px.bar(df_group_cat_topic_FLOP.iloc[:10,:], x=df_group_cat_topic_FLOP.index[:10], y='count')
        
    return fig, fig_1, fig_2, fig_3, fig_4
    
    



if __name__ == "__main__":
    app.run_server()