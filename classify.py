"""Train-only text vectorization and held-out sentiment evaluation."""
import argparse,json,re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix

def preprocess(text):
    text=re.sub(r'<[^>]+>',' ',str(text).lower())
    text=re.sub(r'https?://\S+',' ',text)
    return re.sub(r'\s+',' ',text).strip()

def prepare(frame):
    if not {'review','sentiment'}<=set(frame.columns):raise ValueError('Expected review and sentiment columns.')
    frame=frame.dropna(subset=['review','sentiment']).copy()
    frame['review']=frame['review'].map(preprocess)
    frame=frame[frame['review'].str.len()>0]
    conflicts=frame.groupby('review')['sentiment'].nunique()
    frame=frame[~frame['review'].isin(conflicts[conflicts>1].index)].drop_duplicates('review')
    if not set(frame.sentiment)<={'positive','negative'} or frame.sentiment.nunique()!=2:raise ValueError('Expected positive and negative labels.')
    return frame

def train_evaluate(frame):
    frame=prepare(frame)
    train,test=train_test_split(frame,test_size=.2,random_state=42,stratify=frame.sentiment)
    assert not set(train.review)&set(test.review)
    reports={}
    for name,estimator in [('LogisticRegression',LogisticRegression(max_iter=1000)),('MultinomialNB',MultinomialNB())]:
        model=Pipeline([('vectorizer',CountVectorizer(stop_words='english',max_features=30000)),('model',estimator)])
        model.fit(train.review,train.sentiment)
        pred=model.predict(test.review)
        vectorizer=model.named_steps['vectorizer']
        # Real corpus frequencies, rather than counting each vocabulary entry once.
        counts=vectorizer.transform(train.review).sum(axis=0).A1
        vocab=vectorizer.get_feature_names_out()
        top=counts.argsort()[-15:][::-1]
        reports[name]={'accuracy':float(accuracy_score(test.sentiment,pred)),
          'macro_f1':float(f1_score(test.sentiment,pred,average='macro')),
          'classification_report':classification_report(test.sentiment,pred,output_dict=True,zero_division=0),
          'confusion_matrix':confusion_matrix(test.sentiment,pred,labels=model.classes_).tolist(),
          'top_training_words':{str(vocab[i]):int(counts[i]) for i in top}}
    return {'dataset':'IMDB supplied CSV','seed':42,'train_rows':len(train),'test_rows':len(test),
            'duplicates_and_conflicts_removed':True,'train_test_text_overlap':0,'models':reports}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('metrics.json'))
    a=p.parse_args();report=train_evaluate(pd.read_csv(a.data));a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v['accuracy'] for k,v in report['models'].items()}))

if __name__=='__main__':main()
