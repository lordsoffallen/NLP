from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import nltk
import pandas as pd
import re
import string


stopwords = nltk.corpus.stopwords.words('english')
lm = nltk.WordNetLemmatizer()

data = pd.read_csv("SMSSpamCollection.tsv", sep='\t')
data.columns = ['label', 'body_text']

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [lm.lemmatize(word) for word in tokens if word not in stopwords]
    return text

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

# Feature engineering: Creating two variables:
# punctuation percentage of the text and text char size
data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))

X_train, X_test, y_train, y_test = train_test_split(data[['body_text', 'body_len', 'punct%']], 
                                                    data['label'], test_size=0.2)

tfidf = TfidfVectorizer(analyzer=clean_text)
tfidf.fit(X_train['body_text'])

tfidf_train = tfidf.transform(X_train['body_text'])
tfidf_test = tfidf.transform(X_test['body_text'])

X_train_vect = pd.concat([X_train[['body_len', 'punct%']].reset_index(drop=True), 
                          pd.DataFrame(tfidf_train.toarray())], axis=1)
X_test_vect = pd.concat([X_test[['body_len', 'punct%']].reset_index(drop=True), 
                         pd.DataFrame(tfidf_test.toarray())], axis=1)

# RANDOM FOREST
rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)

rf_model = rf.fit(X_train_vect, y_train)
y_pred = rf_model.predict(X_test_vect)

print(classification_report(y_test, y_pred))

############################################################
#                  precision    recall  f1-score   support
#
#          ham       0.98      1.00      0.99       965
#         spam       1.00      0.86      0.92       149
#
#    micro avg       0.98      0.98      0.98      1114
#    macro avg       0.99      0.93      0.96      1114
# weighted avg       0.98      0.98      0.98      1114
#############################################################

# GRADIENT BOOSTING
gb = GradientBoostingClassifier(n_estimators=150, max_depth=11)

gb_model = gb.fit(X_train_vect, y_train)
y_pred = gb_model.predict(X_test_vect)

print(classification_report(y_test, y_pred))

############################################################
#                  precision    recall  f1-score   support
#
#          ham       0.98      0.99      0.98       965
#         spam       0.93      0.85      0.89       149
#
#    micro avg       0.97      0.97      0.97      1114
#    macro avg       0.96      0.92      0.94      1114
# weighted avg       0.97      0.97      0.97      1114
#############################################################