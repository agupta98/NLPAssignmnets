# coding: utf-8
"""CS585: Assignment 4
See README.md

"""

### DO NOT ADD TO THESE IMPORTS. ####
from collections import Counter
from itertools import product
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import urllib.request
from gensim.models import Word2Vec
from nltk.corpus import brown
#####################################


def download_data():
    """ Download labeled data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/bqitsnhk911ndqs/train.txt?dl=1'
    urllib.request.urlretrieve(url, 'train.txt')
    url = 'https://www.dropbox.com/s/s4gdb9fjex2afxs/test.txt?dl=1'    
    urllib.request.urlretrieve(url, 'test.txt')
    
    
def read_data(filename):
    """
    Read the data file into a list of lists of tuples.
    
    Each sentence is a list of tuples.
    Each tuple contains four entries:
    - the token
    - the part of speech
    - the phrase chunking tag
    - the named entity tag
    
    For example, the first two entries in the
    returned result for 'train.txt' are:
    
    > train_data = read_data('train.txt')
    > train_data[:2]
    [[('EU', 'NNP', 'I-NP', 'I-ORG'),
      ('rejects', 'VBZ', 'I-VP', 'O'),
      ('German', 'JJ', 'I-NP', 'I-MISC'),
      ('call', 'NN', 'I-NP', 'O'),
      ('to', 'TO', 'I-VP', 'O'),
      ('boycott', 'VB', 'I-VP', 'O'),
      ('British', 'JJ', 'I-NP', 'I-MISC'),
      ('lamb', 'NN', 'I-NP', 'O'),
      ('.', '.', 'O', 'O')],
     [('Peter', 'NNP', 'I-NP', 'I-PER'), ('Blackburn', 'NNP', 'I-NP', 'I-PER')]]
    """
    ###TODO
    with open(filename,'r') as f:
        words = f.readlines()
    li1=[]
    li2=[]
    words = [x.strip('\n') for x in words]
    for i in words:
        if (i=='-DOCSTART- -X- -X- O'):
            continue
        if (i !=''):
            i = i.split(' ')
            li1.append(tuple(i))
        if (i ==''):
            li2.append(li1)
            li1=[]
    li2.append(li1)
    li1=[]
    data=list(filter(None,li2))
    
    return data

        
        
        
    

def make_feature_dicts(data,w2v_model,token=True,
                       caps=True,pos=True,
                       chunk=True,
                       context=True,w2v=True):
    """
    Create feature dictionaries, one per token. Each entry in the dict consists of a key (a string)
    and a value of 1.
    Also returns a numpy array of NER tags (strings), one per token.

    See a3_test.

    The parameter flags determine which features to compute.
    Params:
    data.......the data returned by read_data
    token......If True, create a feature with key 'tok=X', where X is the *lower case* string for this token.
    caps.......If True, create a feature 'is_caps' that is 1 if this token begins with a capital letter.
               If the token does not begin with a capital letter, do not add the feature.
    pos........If True, add a feature 'pos=X', where X is the part of speech tag for this token.
    chunk......If True, add a feature 'chunk=X', where X is the chunk tag for this token
    context....If True, add features that combine all the features for the previous and subsequent token.
               E.g., if the prior token has features 'is_caps' and 'tok=a', then the features for the
               current token will be augmented with 'prev_is_caps' and 'prev_tok=a'.
               Similarly, if the subsequent token has features 'is_caps', then the features for the
               current token will also include 'next_is_caps'.
    Returns:
    - A list of dicts, one per token, containing the features for that token.
    - A numpy array, one per token, containing the NER tag for that token.
    """
    ###TODO
    '''
    li=[]
    lables =[]
    for line in data:
        dicts =[]
        for word in line:
            dict={}
            if (token == True):
                dict['tok='+word[0].lower()]=1
            if (caps == True):
                if (word[0][0].isupper()):
                    dict['is_caps']=1
            if (pos==True):
                dict['pos='+word[1]]=1
            if (chunk == True):
                dict['chunk='+word[2]]=1
            dicts.append(dict)
            lables.append(word[3])
            dict={}
        li.append(dicts)
    lables=np.asarray(lables)
    
    if (context == True) and (token==True):
        dicts =[]
        for sentence in li:
            n= len(sentence)
            for i in range(len(sentence)):
                currDict = {}
                for k in sentence[i].keys():
                    currDict[k] = 1
                if (i==0) and (n==1):
                    currDict[k] = 1
                if (i==0) and n >1:
                    nextDict = sentence[i+1]
                    for k in nextDict.keys():
                        currDict['next_'+k]=1
                elif (i==n-1) and n > 1:
                    prevDict =sentence[i-1]
                    for k in prevDict.keys():
                        currDict['prev_'+k]=1
                elif (i>0) and (i<n-1):
                    
            
                    nextDict = sentence[i+1]
                    for k in nextDict.keys():
                        currDict['next_'+k]=1
                    prevDict =sentence[i-1]
                    for k in prevDict.keys():
                        currDict['prev_'+k]=1
                
                dicts.append(currDict)
    else:
        dicts =[]
        for i in li:
            for j in i:
                dicts.append(j)
            

    #print(dicts)    
    return dicts,lables
    '''
    all_dicts =[]
    all_labels =[]
    for sentence in data:
        sentence_dicts = []
        for i,word in enumerate(sentence):
            feats ={}
            if token:
                feats['tok=%s' %word[0].lower()] = 1
            if caps and word[0][0].isupper():
                feats['is_caps'] = 1
            if pos:
                feats['pos=%s' % word[1]] = 1
            if chunk:
                feats['chunk=%s' %word[2]] = 1
            if w2v:
                try:
                    for i,vector in enumerate(w2v_model.wv[word[0]]):
                        feats['w2v_%s'%str(i+1)] = vector
                except:
                    pass
            sentence_dicts.append(feats)
            all_labels.append(word[-1])
        if context:
            new_dicts = []
            for i,d in enumerate(sentence_dicts):
                new_dict = dict(d)
                if i > 0:
                    for k,v in sentence_dicts[i-1].items():
                        new_dict['prev_%s' %k] = v
                        
                if i < len(sentence_dicts) - 1:
                    for k,v in sentence_dicts[i+1].items():
                        new_dict['next_%s' %k] = v
                        
                        
                new_dicts.append(new_dict)
                
            sentence_dicts = new_dicts
        all_dicts.extend(sentence_dicts)
    #print(all_dicts)
    #print(all_labels)
    return all_dicts,np.array(all_labels)        
                    


def confusion(true_labels, pred_labels):
    
    
    """
	Create a confusion matrix, where cell (i,j)
	is the number of tokens with true label i and predicted label j.

	Params:
	  true_labels....numpy array of true NER labels, one per token
	  pred_labels....numpy array of predicted NER labels, one per token
	Returns:
	A Pandas DataFrame (http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
	See Log.txt for an example.
	"""
	###TODO
    
   #df_ =pd.DataFrame(index=true_labels,columns=pred_labels)
   #return df_
    true_lables_index =['I-LOC','I-MISC','I-ORG','I-PER','O']
    pred_lables_col =['I-LOC','I-MISC','I-ORG','I-PER','O']
    a = np.zeros(shape=(len(true_lables_index),len(pred_lables_col)))
    dataframe = pd.DataFrame(data = a , index = true_lables_index,columns = pred_lables_col)
    #print("helllloooo")
    for i,j in zip(true_labels,pred_labels):
        dataframe[j][i] +=1
    
    return dataframe
	

def evaluate(confusion_matrix):
    """
	Compute precision, recall, f1 for each NER label.
	The table should be sorted in ascending order of label name.
	If the denominator needed for any computation is 0,
	use 0 as the result.  (E.g., replace NaNs with 0s).

	NOTE: you should implement this on your own, not using
	      any external libraries (other than Pandas for creating 
	      the output.)
	Params:
	  confusion_matrix...output of confusion function above.
	Returns:
	  A Pandas DataFrame. See Log.txt for an example.
	"""
    confusion_matrix = confusion_matrix.fillna(0)
    precision=[]
    for i in range(len(confusion_matrix.index)):
        n=confusion_matrix.index[i]
        n1=confusion_matrix.columns[i]
        precision.append(confusion_matrix[n][n1]/confusion_matrix.sum(axis=0)[i])
    recall=[]
    for i in range(len(confusion_matrix.index)):
        n=confusion_matrix.index[i]
        n1=confusion_matrix.columns[i]
        recall.append(confusion_matrix[n][n1]/confusion_matrix.sum(axis=1)[i])
    f1=[]
    for i,j in zip(precision,recall):
        f1.append((2*i*j)/(i+j))
    li1=[precision,recall,f1]
    df1=pd.DataFrame(li1,index=['precision','recall','f1'],columns=['I-LOC','I-MISC','I-ORG','I-PER','O'])
    return df1

def average_f1s(evaluation_matrix):
    """
	Returns:
	The average F1 score for all NER tags, 
	EXCLUDING the O tag.
	"""
	###TODO
    df1=evaluation_matrix[2:]
    del df1['O']
    avg = df1.sum(axis=1)/len(df1.columns)
    avg=avg.loc['f1']
    #li =[avg]
    #print(avg)
    return avg
    
    
    

def evaluate_combinations(train_data, test_data):
    """
	Run 16 different settings of the classifier, 
	corresponding to the 16 different assignments to the
	parameters to make_feature_dicts:
	caps, pos, chunk, context
	That is, for one setting, we'll use
	token=True, caps=False, pos=False, chunk=False, context=False
	and for the next setting we'll use
	token=True, caps=False, pos=False, chunk=False, context=True

	For each setting, create the feature vectors for the training
	and testing set, fit a LogisticRegression classifier, and compute
	the average f1 (using the above functions).

	Returns:
	A Pandas DataFrame containing the F1 score for each setting, 
	along with the total number of parameters in the resulting
	classifier. This should be sorted in descending order of F1.
	(See Log.txt).

	Note1: You may find itertools.product helpful for iterating over
	combinations.

	Note2: You may find it helpful to read the main method to see
	how to run the full analysis pipeline.
	"""
	###TODO
    a = list(product([True,False],repeat=5))
    download_data()
    train_data = read_data('train.txt')
    test_data =read_data('test.txt')
    li2=[]
    for i in a:
        li1=[]
        dicts,labels=make_feature_dicts(train_data,w2v_model, token=True, caps=i[0], pos=i[1], chunk=i[2], context=i[3],w2v=i[4])
        vec = DictVectorizer()
        X = vec.fit_transform(dicts)
        clf = LogisticRegression()
        clf.fit(X,labels)
        test_dicts,test_labels = make_feature_dicts(test_data,w2v_model,token=True,caps=i[0], pos=i[1], chunk=i[2],
                                                    context=i[3],w2v=i[4])
        X_test = vec.transform(test_dicts)
        preds = clf.predict(X_test)
        confusion_matrix = confusion(test_labels, preds)
        evaluation_matrix = evaluate(confusion_matrix)
        f11 = average_f1s(evaluation_matrix)
        #print(f11)
        n_params = clf.coef_.size
        li1.extend([f11,n_params,i[0],i[1],i[2],i[3],i[4]])
        li2.append(li1)
        
    
        
    #return li2
    df1=pd.DataFrame(li2,index=list(range(32)),columns=['f1','n_params','caps','pos','chunk','context','w2v'])
    df1=df1.sort_values(by='f1',ascending=False)
    return df1
    
        
        
        
        
        
        
        
    
    
    
    

if __name__ == '__main__':
    sents = brown.sents()
    w2v_model = Word2Vec(sents,size =50,window =5,min_count =5)
    download_data()
    train_data = read_data('train.txt')
    dicts, labels = make_feature_dicts(train_data,w2v_model,
                                   token=True,
                                   caps=True,
                                   pos=True,
                                   chunk=True,
                                   context=True, w2v= True)
    vec = DictVectorizer()
    X = vec.fit_transform(dicts)
    print('training data shape: %s\n' % str(X.shape))
    clf = LogisticRegression()
    clf.fit(X, labels)
    test_data = read_data('test.txt')
    test_dicts, test_labels = make_feature_dicts(test_data,w2v_model,
	                                             token=True,
	                                             caps=True,
	                                             pos=True,
	                                             chunk=True,
	                                             context=True, w2v =True)            
    X_test = vec.transform(test_dicts)
    print('testing data shape: %s\n' % str(X_test.shape))
    preds = clf.predict(X_test)
    confusion_matrix = confusion(test_labels, preds)
    print('confusion matrix:\n%s\n' % str(confusion_matrix))
    evaluation_matrix = evaluate(confusion_matrix)
    print('evaluation matrix:\n%s\n' % str(evaluation_matrix))
    print('average f1s: %f\n' % average_f1s(evaluation_matrix))
    combo_results = evaluate_combinations(train_data, test_data)
    print('combination results:\n%s' % str(combo_results))
    '''
    test_dicts, test_labels = make_feature_dicts(data,w2v_model,
	                                             token=True,
	                                             caps=True,
	                                             pos=True,
	                                             chunk=True,
	                                             context=True,w2v=True) 
    print(test_dicts, test_labels)
    '''