from flask import Flask,request,render_template
#import nltk
from nltk import sent_tokenize,word_tokenize,PorterStemmer
from nltk.corpus import stopwords
stopWords = set(stopwords.words("english"))
ps = PorterStemmer()
import math

app=Flask(__name__)

def frequency_table(text):
    words=word_tokenize(text)
    word_freq=dict()
    for x in words:
        x=ps.stem(x.lower())
        if (x not in stopWords) and (len(x)>2):
            if x not in word_freq.keys():
                word_freq[x]=1
            else:
                word_freq[x]+=1
    return word_freq

def frequency_matrix(sentences):
    frequency_matrix=dict()
    for sent in sent_tokenize(sentences):
        frequency_matrix[sent[:15]]=frequency_table(sent)
    return frequency_matrix

def tf_matrix(freq_matrix): 
    tf_matrix={}
    for sent,freq_table in freq_matrix.items():
        tf_table={}
        ft_len=len(freq_table)
        for Dict in freq_table.keys():
            tf_table[Dict]=round(freq_table[Dict]/ft_len,2)
        tf_matrix[sent]=tf_table
    return tf_matrix

def doc_frequency(freq_matrix):
    doc_table={}
    for sent,tf_table in freq_matrix.items():
        for word in tf_table.keys():
            if word not in doc_table.keys():
                doc_table[word]=1
            else:
                doc_table[word]+=1
    return doc_table

def idf_matrix(freq_matrix,doc_per_word):
    total_doc=len(freq_matrix.keys())
    tf_matrix={}
    for sent,tf_table in freq_matrix.items():
        freq_table={}
        for word in tf_table.keys():
            freq_table[word]=round(math.log10(total_doc/doc_per_word[word]),2)
        tf_matrix[sent]=freq_table
    return tf_matrix

def tf_idf_matrix(tf_matrix,idf_matrix):
    tf_idf_matrix={}
    for (sent1,ft_table1),(sent2,ft_table2) in zip(tf_matrix.items(),idf_matrix.items()):
        freq_table={}
        for (word1,value1),(word2,value2) in zip(ft_table1.items(),ft_table2.items()):
                freq_table[word1]=round(value1*value2,2)
        tf_idf_matrix[sent1]=freq_table
    return tf_idf_matrix

def score_sentence(tf_idf_matrix):
    sent_score={}
    for sent,ft_table in tf_idf_matrix.items():
        for word,value in ft_table.items():
            if sent not in sent_score.keys():
                sent_score[sent]=value
            else:
                sent_score[sent]+=value
                
    return sent_score

def average_score(sent_score):
    sum_sent=0
    for value in sent_score.values():
        sum_sent+=value
    avg_score=round(sum_sent/len(sent_score),2)
    return avg_score

def get_summary(sentences,sentence_score,threshold):
    summary=""
    for sent in sentences:
        if (sent[:15] in sentence_score) and sentence_score[sent[:15]]>=threshold:
                summary+=" "+sent
    return summary

def lets_summarize(text):
#     summary=""
    sentences=sent_tokenize(text)
    
    freq_matrix=frequency_matrix(text)
    
    tf_matrix1=tf_matrix(freq_matrix)
    
    doc_matrix=doc_frequency(freq_matrix)
    
    idf1_matrix=idf_matrix(freq_matrix,doc_matrix)
    
    tf_idf_matrix1=tf_idf_matrix(tf_matrix1,idf1_matrix)
    
    sent_score=score_sentence(tf_idf_matrix1)
    
    threshold=average_score(sent_score)
    
    summary=get_summary(sentences,sent_score,1.15*threshold)
    return summary


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    raw_text=request.form['paragraph_text']
    summary=lets_summarize(raw_text)
    
    
    return render_template('index.html',prediction_text=summary)

if __name__ == '__main__':
    app.run(debug=True)