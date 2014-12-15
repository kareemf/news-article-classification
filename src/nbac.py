'''
Name: Kareem Francis
Assignment id: AIAF
Due Date: May. 27, 2012
Created on May 3, 2012
'''

import os
import math
import nltk
import string
import argparse
import cPickle as pickle
from article import *
from numpy import array
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tag.simplify import simplify_wsj_tag
from nltk.cluster import KMeansClusterer, cosine_distance

class NaiveBayesArticleClassifier(object):
    '''Naive Bayes Article Classifier'''

    def __init__(self,training_dir,training_file_name, verbose_mode=True):
        '''
        Read in, pre-process, and construct training data from specified path, set up vectors used in computation
        '''
 
        self.verbose_mode = verbose_mode
        self.training_dir = training_dir
        self.training_path = training_file_name
        
        self.model = None
        
        self.categories = [] #list of categories will be constructed dynamically from labeled data
        self.keywords = []#list of every unique key word used across all training_articles - could probably just use keyword_count.keys?
        self.class_count = defaultdict(give_me_0) #count occurrences of each category
        self.key_count = defaultdict(give_me_0) #keyword frequency count
        self.key_class = defaultdict(dd0)# keyword given class frequency count [][]
        
        self.training_articles = []#list of all labeled training_articles
        
        self.verbose('reading stop words')
        with open('stopwords.txt','r') as sw_file:
            stopword_list = [line.strip().lower() for line in sw_file]
            
        #builds a dictionary, which maps letters to a list of stopwords which begin with that letter
        self.verbose('initializing stop word vectors')
        self.stopdict = {}
        for i in xrange(26):
            char = chr(i + 97) #'a' = char 97
            self.stopdict[char] = [word for word in stopword_list if  word[0] == char]
        
        self.verbose('initializing data vectors')
        with open(training_file_name,'r') as training_file:
            for line in training_file:
                article_id,classification = line.strip().split(',') 
                classification = classification.strip()  
                
                #learn classification labels
                if classification.strip() not in self.categories:
                    self.categories.append(classification.strip()) 
                    
                self.class_count[classification]+=1
    
                article_path = os.path.join(training_dir,article_id+'.txt')
                content = self.parse_article(article_path)  
                for word in content:
                    if word not in self.keywords:
                        self.keywords.append(word) #new feature
                    self.key_count[word]+=1
                    self.key_class[word][classification]+=1
                    
                self.training_articles.append(Article(article_id,classification,content))
                
        self.verbose('vectors successfully initialized')
        
    def train(self): 
        '''Train a Bayes model using add-one smoothing'''
        
        n = len(self.training_articles) #number of training documents
        prior = {} #p(C) -> prior probability (weight) of a class
        likilihood = defaultdict(dd0) #p(w|C) = w/sum(w) -> likelihood (weight) of a word given a class
        
        self.verbose('beginning model training')
        for category in self.categories:
            prior[category] =  self.class_count[category]/float(n) 
            self.verbose('\tprior(%s): %r' % (category, prior[category]))
            #all words in class
            words_in_category = []
            #all articles in class
            articles_of_category = [article for article in self.training_articles if article.classification == category]    
            for article in articles_of_category:
                words_in_category.extend(article.content)
                
            self.verbose('\twords in category %s: %i \n' % (category,len(words_in_category)))
            
            #calculate conditional probability of word given class:
            #adding one to m keywords = m. total words in category = |words in category|
            total = len(words_in_category) + len(self.keywords)
            for word in self.keywords:
                #add one -> minimum key_class value of one
                likilihood[word][category] = (self.key_class[word][category]+1)/float(total)  
        
        #classification model
        self.model = prior, likilihood
        self.verbose('model training successful')
        return self.model
    
    def test(self, article):
        '''Apply model to test article'''
       
        prior, likilihood = self.model
        most_probable = None
        highest_probability = 0.0
        
        self.verbose('article: %s' % article.id)
        #probability of class given article = posterior(c|a)
        for category in self.categories:
            #log_b(x*y) = log_b (x) + log_b (y) -> avoid infinitesimal small values 
            posterior = math.log(prior[category] + 1)
            #adding 1 to avoid negative values for log(x<1)
            for word in article.content:
                posterior += math.log(likilihood[word][category] + 1)
            
            self.verbose("p(%s|%s) = %r" % (category, article.id, posterior))
            if posterior > highest_probability:
                highest_probability = posterior
                most_probable = category
        
        return most_probable

    def cluster(self, k=5, repeats=1):
        '''
        Cluster documents into k clusters using the NLTK
        implementation of K-Means clustering. The frequency of each
        unique word across an article serves as its feature vector.
        '''
        article_freq_count = {} #frequency of each unique word in a given article
        for article in self.testing_articles:
            article_freq_count[article.id] = []
            for unique_word in self.keywords:
                #count frequency of word in article, add to frequency list
                article_freq_count[article.id].append(article.content.count(unique_word))

        #nltk k-means requires numpy array-like objects
        vectors = [array(article_freq_count[article]) for article in article_freq_count]
        clusterer = KMeansClusterer(k, cosine_distance, repeats=repeats)
        clusters = clusterer.cluster(vectors, True, trace=False)

        groups = [[] for _ in xrange(k)]

        #vector positions need to be converted back to article IDs,
        #because IDs are striped during vector construction.
        vector_ids = {} #maps positions in the vector to article IDs
        f =  article_freq_count.copy()
        for pos in xrange(len(vectors)):
            for id in f.keys():
                #equivalent to 'if article_freq_count[id] == vectors[pos]',
                #but numpy equivalence checking is weird
                t = article_freq_count[id] == vectors[pos]
                if not False in t:
                    vector_ids[pos] = id
                    f.pop(id)

        for i in xrange(len(clusters)):
            groups[clusters[i]].append(vector_ids[i])

        return groups

    def run_tests(self, testing_dir, testing_file):   
        '''Construct the set of test articles, perform evaluation, and re '''
        
        testing_articles = [] #the set of training articles
        self.predictions = [] #set of test predictions
        overall_correct = 0 #overall correct
        correct = defaultdict(give_me_0) #correct per class
        occurrences= defaultdict(give_me_0) #occurrences per class
        class_accuracy = {} #accuracy per class
        
        self.verbose('initialzing test article set')
        #construct test article set
        with open(testing_file,'r') as testing_file:
            for line in testing_file:
                article_id,classification = line.strip().split(',')
                classification = classification.strip()  
                
                #should not be encountering any new class values at this point                                                
                if classification not in self.categories:
                    raise Exception('testing data is misclassified')
     
                try:
                    article_path = os.path.join(testing_dir,article_id + '.txt')
                    raw_content = self.parse_article(article_path)  
                    content = [] 
                    #only consider words in keyword vocabulary, otherwise, they have not been learned       
                    for word in raw_content:    
                        if word in self.keywords:
                            content.append(word)
                    testing_articles.append(Article(article_id,classification,content))
                    
                except IOError:
                    print 'Error opening %s, make sure file an folder exist.' % article_path
                except Exception, e:
                    print e

        self.testing_articles = testing_articles
        self.verbose('performing tests')          
        for article in testing_articles:        
            prediction = self.test(article)
            self.predictions.append([article.id, article.classification, prediction])
            self.verbose('actual class: %s, predicted class: %s' %(article.classification, prediction))
            
            occurrences[article.classification]+=1
            if prediction == article.classification:
                overall_correct+=1
                correct[article.classification]+=1
            
        for category in self.categories:
            class_accuracy[category] = 100 *( correct[category]/float(occurrences[category]))
            self.verbose('\taccuracy[%s]: %r' % (category, class_accuracy[category]))
        
        overall_accuracy = 100 * (float(overall_correct)/len(testing_articles))
        
        self.verbose('testing phase completed') 
        return overall_accuracy, class_accuracy
    
    def parse_article(self, article_path):
        '''
        Read an article in from file path, return the preprocessed list of words in the article.
        Pre-processing performs natural language processing on the article, removing "stop words."
        '''
        
        content = [] #article is modeled as a set of words/tokens, as opposed to a string           
        with open(article_path,'r') as article:
            for line in article:
                content.extend(self.preprocess(line)) #reduce words only to essential keywords    
        return content
    
    def preprocess(self, text='',method=0):
        '''
        Given some text, return a list of words/tokens, after removing punctuations,
        stop words, and digits. There various methods that may be used, depending on the
        method flag, including part of speech tagging, and checking against various stop word
        lists
        '''

        #remove punctuation and digits
        out = text.translate(string.maketrans("",""), string.punctuation + string.digits)
        #return word tokens in the line
        tokenized_words = nltk.word_tokenize(out.lower())

        if method == 0:
            #indexes stopdict by first char of word if first char = letter
            keywords = [word for word in tokenized_words 
                        if ord(word[0]) in xrange(97,123) and word not in self.stopdict[word[0]]]

        elif method == 1:
            #use NLTK stopword list
            keywords = [word for word in tokenized_words
                        if word not in stopwords.words('english')]

        elif method == 2:
            #only keep words that correspond to desired parts of speech.
            #only want to consider: nouns, proper nouns,and verbs
            #(presentense, pastence verb, present participle, past participle

            tokenized_words = nltk.word_tokenize(text.lower())
            #assign part of speech to each word in text
            pos_tagged_words = nltk.pos_tag(tokenized_words)
            #use simplified tagging for less parts of speech
            simplified_tagged_text = [(word, simplify_wsj_tag(tag)) for word, tag in pos_tagged_words]
        
            keywords = []
            for (word,part_of_speech) in simplified_tagged_text:    
                whitelist = ['N','NP', 'V', 'VD', 'VG', 'VN']
                if part_of_speech in whitelist:
                    keywords.append(word)
                
        return keywords
    
    def write_predictions(self, out_file):
        '''Write predictions of test article labels to specified file'''
        try:
            with open(out_file, 'wb') as of:
                of.write('article id, actual classification, predicted classification \n')
                for prediction in self.predictions:
                    of.write('%s: %s, %s\n' % (prediction[0], prediction[1], prediction[2]))
        except Exception, e:
            print 'error writing predictions to file: ', e

    def verbose(self, message = ''):
        '''Print messages only if flag is true'''
        if self.verbose_mode:
            print message

def give_me_0(): return 0
def give_me_1(): return 1
def dd0(): return defaultdict(give_me_0)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Naive Bayes Article Classifier')
    parser.add_argument('-training_dir',
                        help='A directory containing training articles and a class.txt file.',
                        default=os.path.join('.'+os.curdir,'training_articles'))
    parser.add_argument('-testing_dir',
                        help='A directory containing test articles and a class.txt file',
                        default=os.path.join('.'+os.curdir,'testing_articles'))
    parser.add_argument('-o', '--output',
                        help='The file to which output is to be written. Defaults to predictions.txt in current directory',
                        default='predictions.txt')
    parser.add_argument('-c', '--clean',
                        help='Execute without using previous data. Necessary if training data has changed',
                        action='store_true')
    parser.add_argument('-v', '--verbose',
                        help='Execute in verbose mode. Greatly increases printed output.',
                        action='store_true')
    args = parser.parse_args()
        
    clean = args.clean
    out_file = args.output
    verbose_mode = args.verbose    
    training_dir = args.training_dir
    testing_dir  = args.testing_dir
    training_file = os.path.join(training_dir,'class.txt')
    testing_file = os.path.join(testing_dir,'class.txt')
    
    data_path = 'data.p'
    model_path = 'model.p'
    
    if os.path.isfile(data_path) and not clean:
        #look for saved data from previous session
        try:
            print 'found existing data'
            with open(data_path, 'rb') as data_pickle:
                nbac = pickle.load(data_pickle)
                nbac.verbose_mode = verbose_mode
            print 'data loaded'
        except Exception, e:
            print 'error unpickling data: ', e
    else:
        try:
            print 'initializing...'
            nbac = NaiveBayesArticleClassifier(training_dir,training_file,verbose_mode)
            with open(data_path, 'wb') as data_pickle:
                pickle.dump(nbac, data_pickle)
            print 'data successfully serialized'
        except Exception,e:
            #failing to pickle doesn't halt execution (naturally)
            print 'error computing/pickling data:' , e

    if os.path.isfile(model_path) and not clean:
        try:
            #may want to use same training model on different testing data
            print 'found existing model'
            with open(model_path, 'rb') as model_pickle:
                nbac.model = pickle.load(model_pickle)
            print 'model loaded'
        except Exception, e:
            print 'error unpickling model: ', e
    else:
        try:
            print 'constructing model...'
            model = nbac.train()
            with open(model_path, 'wb') as model_pickle:
                pickle.dump(model, model_pickle)
            print 'model successfully serialized'
        except Exception,e:
            print 'error computing/pickling model:' , e

     
    print 'beginning testing phase...'
    acc, class_acc = nbac.run_tests(testing_dir, testing_file)
    
    print 'writing predictions to file'
    nbac.write_predictions(out_file)
    print 'successfully wrote predictions to file'
    
    print 'overall accuracy: ', acc
    for c in nbac.categories:
        print 'accuracy[%s]: %r' % (c, class_acc[c])

    print 'clustering documents...'
    groups = nbac.cluster()
    for i in xrange(len(groups)):
        print "Group %i: %s " % (i, groups[i])

    try:
        with open('clusters.txt','w') as cluster_file:
            for i in xrange(len(groups)):
                cluster_file.write("Group %i: %s \n" % (i, groups[i]))
    except Exception, e:
        print 'error writing predictions to file: ', e
    print 'successfully wrote clusters to file'
