# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:58:33 2020

@author: AKSHATA
"""

#import all neccessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as rd

#read the csv file
df = pd.read_csv('clustering.csv')
df.head()

#for clustering, taking only two features out of whole dataset
x = df[['LoanAmount','ApplicantIncome']]
plt.scatter(x["ApplicantIncome"],x["LoanAmount"], c='black')
plt.ylabel('Loan Amount')
plt.xlabel('Applicant Income')
plt.show()

#define number of clusters
K = 3

#select random observations for centroid
centroids = (x.sample(n=K))
plt.scatter(x["ApplicantIncome"],x["LoanAmount"], c='black')
plt.scatter(centroids["ApplicantIncome"],centroids["LoanAmount"], c='red')
plt.ylabel('Loan Amount')
plt.xlabel('Applicant Income')
plt.show()

#Assigning all points to the centroid
diff = 1
j=0

while (diff!=0):
    XD = x
    i = 1
    for index1,row_c in centroids.iterrows():
        ED = []
        for index2,row_d in XD.iterrows():
            d1 = (row_c['ApplicantIncome'] - row_d['ApplicantIncome'])**2
            d2 = (row_c['LoanAmount'] - row_d['LoanAmount'])**2
            d = np.sqrt(d1+d2)
            ED.append(d)
            #print(ED)
        x[i] = ED
        i  = i+1
      
    C = []
    for index,row in x.iterrows():
        min_dist = row[1]
        pos =1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos = i+1
        C.append(pos)
        
    x['Cluster'] = C
    centroids_new = x.groupby(["Cluster"]).mean()[['LoanAmount','ApplicantIncome']]
    if j == 0:
        diff = 1
        j = j+1
    else:
        diff = (centroids_new['LoanAmount'] - centroids['LoanAmount']).sum() + (centroids_new['ApplicantIncome'] - centroids['ApplicantIncome']).sum()
        print(diff.sum())
    centroids = x.groupby(["Cluster"]).mean()[["LoanAmount","ApplicantIncome"]]
   
#visualize the output
color = ['blue','green','cyan']

for k in range(K):
    data = x[x['Cluster'] == k+1]
    plt.scatter(data["ApplicantIncome"],data["LoanAmount"],c=color[k])
    
plt.scatter(centroids["ApplicantIncome"],centroids["LoanAmount"],c='red')
plt.xlabel('Income')
plt.ylabel('Loan Amount (In Thousands)')
plt.show()
       