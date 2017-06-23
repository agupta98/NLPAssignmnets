# coding: utf-8
"""CS585: Assignment 2

In this assignment, you will complete an implementation of
a Hidden Markov Model and use it to fit a part-of-speech tagger.
"""

from collections import Counter, defaultdict
import math
import numpy as np
import os.path
import urllib.request


class HMM:
    
    def __init__(self, smoothing=0):
        
        self.smoothing = smoothing
        
    
        

    def fit_transition_probas(self, tags):
        
        
        d = defaultdict(lambda:Counter())
        
        
        
            
        li =[]
        for i in tags:
            for k in range(0,len(i) - 1):
                li.append([i[k],i[k+1]])
        #print(li)
        for i in li:
            d[i[0]][i[1]] +=1.0
        self.states =[]
        for k in sorted(d):
            self.states.append(k)
             
        for k in d.keys():
            s = sum(d[k].values())
            
            for i in d.keys():
                #print(len(d))
                d[k][i] =1*((d[k][i] + self.smoothing)/(s + (self.smoothing*len(d))))
            
        self.transition_probas = dict((k,v) for k,v in d.items())
         
    
    def fit_emission_probas(self, sentences, tags):
        
        
        
        
        """
		 Estimate the HMM emission probabilities from the provided data. 

		 Creates a new instance variable called `emission_probas` that is a 
		 dict from a string ('state') to a dict from string to float. E.g.
		 {'N': {'dog': .1, 'cat': .7, 'mouse': 2},
		  'V': {'run': .3, 'go': .5, 'jump': 2},
		  ...
		 }

		 Params:
		 sentences...a list of lists of strings, representing the tokens in each sentence.
		 tags........a list of lists of strings, representing the tags for one sentence.
		 Returns:
		 None		  

		 See test_hmm_fit_emission.
		 """
		 ###TODO
		 
        d=defaultdict(Counter)

        li = list(zip(tags,sentences))
        #print(li)

        for i in li:
            for j in range(len(i[0])):
                d[i[0][j]][i[1][j]] +=1
        li1=[]
        li2=[]
        for i in d.values():
            li1.append(i)
        for j in li1:
            for k in j:
                li2.append(k)
        li2 = list(set(li2))
        len_li2 = len(li2)
        for k in d.keys():
            s=sum(d[k].values())
            for v in li2:
                d[k][v] = d[k][v] + self.smoothing
                den = s + (1*self.smoothing * len_li2)
                d[k][v] =(d[k][v]/den)
                
        self.emission_probas = dict((k,v) for k,v in d.items())
        

         
      
    def fit_start_probas(self, tags):
        """
		 Estimate the HMM start probabilities form the provided data.

		 Creates a new instance variable called `start_probas` that is a 
		 dict from string (state) to float indicating the probability of that
		 state starting a sentence. E.g.:
		 {
			 'N': .4,
			 'D': .5,
			 'V': .1		
		 }

		 Params:
		   tags...a list of lists of strings representing the tags for one sentence.
		 Returns:
			 None

		 See test_hmm_fit_start
		 """
		 ###TODO
        d = Counter()
        li1=[]
        for i in tags:
            for j in i:
                li1.append(j)
        li1 = list(set(li1))
        
        for i in tags:
            d[i[0]] +=1
        #print(d)
        s = sum(d.values())
        len_li = len(li1)
        #for k,v in d.items():
            #d[k] = d[k]/s
        for i in li1:
            d[i] = (d[i] + self.smoothing)/(s + (self.smoothing*len_li))
        self.start_probas = d

        
    def fit(self, sentences, tags):
        
        
        
        """
		 Fit the parameters of this HMM from the provided data.

		 Params:
		   sentences...a list of lists of strings, representing the tokens in each sentence.
		   tags........a list of lists of strings, representing the tags for one sentence.
		 Returns:
			 None		  

		 DONE. This just calls the three fit_ methods above.
		 """
        self.fit_transition_probas(tags)
        self.fit_emission_probas(sentences, tags)
        self.fit_start_probas(tags)


    def viterbi(self, sentence):
        pmv = np.zeros(shape=(len(self.states),len(sentence)))
        bp =np.zeros(shape=(len(self.states),len(sentence)))
        for i in range(len(self.states)):
            pmv[i][0] = self.start_probas[self.states[i]] * self.emission_probas[self.states[i]][sentence[0]]
            bp[i][0]=0
        for i in range(1,len(sentence)):
            for j in range(len(self.states)):
                li=[]
                li1=[]
                for k in range(len(self.states)):
                    li.append(pmv[k][i-1]*self.emission_probas[self.states[j]][sentence[i]]*self.transition_probas[self.states[k]][self.states[j]])
                    li1.append(pmv[k][i-1]*self.transition_probas[self.states[k]][self.states[j]])
                pmv[j][i]= max(li)
                bp[j][i] =li1.index(max(li1))
        li2=[]
        for k in range(len(self.states) - 1):
            li2.append(pmv[k][len(sentence) - 1])
        self.proba=max(li2)
        bptr=li2.index(max(li2))
        li3=[bptr]
        for i in range(len(sentence)-1,0,-1):
            a = bp[int(bptr)][i]
            li3.append(int(a))
            bptr=a
        li3= li3[::-1]
        path=[]
        for i in li3:
            path.append(self.states[i])
            
        return path,self.proba
            
       
        
            
            
	            
	    
	


def read_labeled_data(filename):
    """
	Read in the training data, consisting of sentences and their POS tags.

	Each line has the format:
	<token> <tag>

	New sentences are indicated by a newline. E.g. two sentences may look like this:
	<token1> <tag1>
	<token2> <tag2>

	<token1> <tag1>
	<token2> <tag2>
	...

	See data.txt for example data.

	Params:
	  filename...a string storing the path to the labeled data file.
	Returns:
	  sentences...a list of lists of strings, representing the tokens in each sentence.
	  tags........a lists of lists of strings, representing the POS tags for each sentence.
	"""
	###TODO
    with open(filename,'r') as f:
        words = f.readlines()
    
    words = [x.strip('\n') for x in words]
    sentences=[]
    tags=[]
    li1=[]
    li2=[]
    for i in words:
        i = i.split(' ')
        if (len(i) == 2):
            li1.append(i[0])
            li2.append(i[1])
        if (len(i) !=2):
            sentences.append(li1)
            tags.append(li2)
            li1=[]
            li2=[]
    return sentences,tags
                
	    
	

def download_data():
    """ Download labeled data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/ty7cclxiob3ajog/data.txt?dl=1'
    urllib.request.urlretrieve(url, 'data.txt')

if __name__ == '__main__':
	"""
	Read the labeled data, fit an HMM, and predict the POS tags for the sentence
	'Look at what happened'

	DONE - please do not modify this method.

	The expected output is below. (Note that the probability may differ slightly due
	to different computing environments.)

	$ python3 a2.py  
	model has 34 states
        ['$', "''", ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB', '``']
	predicted parts of speech for the sentence ['Look', 'at', 'what', 'happened']
	(['VB', 'IN', 'WP', 'VBD'], 2.751820088075314e-10)
	"""
	fname = 'data.txt'
	if not os.path.isfile(fname):
		download_data()
	sentences, tags = read_labeled_data(fname)

	model = HMM(.001)
	model.fit(sentences, tags)
	print('model has %d states' % len(model.states))
	print(model.states)
	sentence = ['Look', 'at', 'what', 'happened']
	print('predicted parts of speech for the sentence %s' % str(sentence))
	#print(len(model.states))
	print(model.viterbi(sentence))
	
