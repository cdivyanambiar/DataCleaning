# -*- coding: utf-8 -*-
"""
Data cleaning for IS_Faculty-List.xlsx

# Import Libraries
"""

import numpy as np
import pandas as pd
from wordcloud import WordCloud
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import missingno as msno
from collections import Counter
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

nltk.download('stopwords')
trainfile = "SIS_Faculty-List.xlsx"
df = pd.read_excel(trainfile)

"""# Printing columns"""
print("======================================================================")
print("Columns in the dataset")
print(df.columns)

"""# Renaming big column names to smaller one"""
df.rename(columns = {'DOCUMENT OTHER PROFESSIONAL CERTIFICATION CRITIERA Five Years Work Experience Teaching Excellence Professional Certifications':'ProfessionalCertificationsLastFiveYears', 'Join\nDate': 'JoinDate','Highest\nQualification\nLevel':'HighestQualificationLevel'}, inplace=True)

print("======================================================================")
"""# Printing head of the data frame """
print("Columns in the dataset after renaming ")
print(df.columns)

print("======================================================================")
print("Printing head of data frame after renaming columns")
print(df.head())

print("======================================================================")
"""# Printing tail of the data frame"""
print("Printing tail of data frame after renaming columns")
df.tail()

print("======================================================================")
"""## While printing head and tail we understood the ID column is 0 which will not make any sense in analysis.So removing ID"""
print("While printing head and tail we understood the ID column is 0 which will not make any sense in analysis.So removing ID")
df.pop('ID')

print("======================================================================")
"""# Printing head after removing ID"""
print("Printing head of data frame after deleting the column ID")
df.head()

print("======================================================================")
"""# Identifying columns with null value"""
print("Checking whether column has null values")
df.isna().any()

print("Printing the count of null values")
df.isna().sum()

print("======================================================================")
"""# Checking data types of columns. """
print('Data type of each column of Dataframe :')
dataTypes = df.dtypes
print(dataTypes)

print("======================================================================")
"""# Checking rows with null columns > 4"""
print("Deleting the rows with more than 4 null columns.")
under_threshold_removed = df.dropna(axis='index', thresh=4, inplace=False)
under_threshold_rows = df[~df.index.isin(under_threshold_removed.index)]
print(under_threshold_rows)

print("======================================================================")
"""# Finding unique values of LWD"""
print("Unique value of LWD")
print(df.LWD.unique())

df.fillna(value={'LWD': 'NaT'}, inplace=True)
df.isna().sum()

print("======================================================================")
"""# Improving readability for Grade column"""
print("Unique value of Grade")
print(df.Grade.unique())

print("======================================================================")
print("Improving readability of Grade Column values")
df["Grade"].replace({"FA": "Faculty", "Chair": "Chairman"}, inplace=True)

print(df['Grade'].value_counts())

"""# Plotting missing values using  missingno"""

print("======================================================================")
# Commented out IPython magic to ensure Python compatibility.
print("Plotting missing values with missingno")
msno.matrix(df)

"""# missingno heat map"""
print("Plotting missing values with missingno: heat map")
msno.heatmap(df)

"""# missingno bar"""
print("Plotting missing values with missingno: Bar chart")
msno.bar(df)

print("======================================================================")
"""# Pie chart of Title column"""

print("Plotting all values Title in Bar chart")
print(Counter(df['Title']))

title_type = df.groupby('Title').agg('count')
type_labels = title_type.Name.sort_values().index 
type_counts = title_type.Name.sort_values()

type_counts

plt.figure(1, figsize=(40,50)) 
the_grid = GridSpec(2, 2)
cmap = plt.get_cmap('Spectral')
colors = [cmap(i) for i in np.linspace(0, 1, 8)]
plt.subplot(the_grid[0, 1], aspect=1, title='Departent Vs No.of Employees')
type_show_ids = plt.pie(type_counts, labels=type_labels, autopct='%1.1f%%', shadow=True, colors=colors)
plt.show()

"""# Text Analysis"""

df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)
df = df.fillna('NA')

print("======================================================================")
print("Printing word Cloud for the column HighestQualificationLevel")

"""## Word cloud"""

comment_words = ''
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(['NA', 'public','nan'])
for val in df.HighestQualificationLevel:

    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens) + " "

wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()

print("======================================================================")
print("Vectorizing ProfessionalCertificationsLastFiveYears")
"""## TF-IDF Vectorization"""

def CreateCorpusFromDataFrame():
    corpus = []
    for val in df.ProfessionalCertificationsLastFiveYears:
       corpus.append(str(val))
    return corpus

corpus = CreateCorpusFromDataFrame()

v = TfidfVectorizer()
df_tfidf= pd.DataFrame(v.fit_transform(corpus).toarray(), columns=v.get_feature_names())

print(df_tfidf.head())

print("======================================================================")
"""# Exporting processed data to xlsx"""

file_name = 'SIS_Faculty-List-cleaned.xlsx'
df.to_excel(file_name)
print('DataFrame is written to Excel File successfully.')