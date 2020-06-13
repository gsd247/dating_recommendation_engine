# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:42:32 2020

@author: Gsd
"""
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
ds = pd.read_csv("C:\ML\dating.csv")

my_user = sys.argv[1]
'''names = ds['Name']
genders = ds['Gender']

i=0
for col in genders: 
    if names[i] == my_user:
        print('S')
        my_user_gender = genders[col]
    i+=1
 '''   
#gs = genders.toarray()

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(ds['Interest'])
#age_matrix = tf.fit_transform(ds['Age'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 
results = {}
for idx, row in ds.iterrows():
   similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
   similar_items = [(cosine_similarities[idx][i], ds['Name'][i]) for i in similar_indices] 
   results[row['Name']] = similar_items[1:]
   
def item(id):  
  return ds.loc[ds['Name'] == id]['Interest'].tolist()[0].split(' - ')[0] 
# Just reads the results out of the dictionary. 
#def recommend(item_id, num):
    #print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")   
    #print("-------")    
    #recs = results[item_id][:num]  
    #return recs
    #for rec in recs: 
    #    print("Recommended: " + item(rec[1]) + " (score:" +      str(rec[0]) + ")")
        
#recommend(2,1)

recs = results[my_user]

print(recs[:2])

