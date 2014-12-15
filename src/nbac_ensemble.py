'''
Name: Kareem Francis
Assignment id: AIAF
Due Date: May. 27, 2012
Created on May 16, 2012
'''

import os
import nltk
import math
import argparse
import random
import cPickle as pickle
from article import *
from collections import defaultdict
from nltk.tag.simplify import simplify_wsj_tag

def give_me_0(): return 0
def give_me_1(): return 1
def dd0(): return defaultdict(give_me_0)

class NaiveBayesArticleClassifier(object):
    '''Naive Bayes Article Classifier'''
    
    def __init__(self,training_dir,training_file_name, verbose_mode=True):
        '''
        Read in, pre-process, and construct training data from specified path, set up vectors used in computation
        '''
 
        self.verbose_mode = verbose_mode
        self.training_dir = training_dir
        self.training_path = training_file_name
        
        self.models = []
        self.prior = {}
        
        self.categories = [] #list of categories will be constructed dynamically from labeled data
        self.keywords = []#list of every unique key word used across all training_articles - could probably just use keyword_count.keys?
        class_count = defaultdict(give_me_0) #count occurrences of each category
#        self.key_count = defaultdict(give_me_0) #keyword frequency count
#        self.key_class = defaultdict(dd0)# keyword given class frequency count [][]
        
        self.training_articles = []#list of all labeled training_articles
        
        self.verbose('initializing data vectors')
        with open(training_file_name,'r') as training_file:
            for line in training_file:
                article_id,classification = line.strip().split(',') 
                classification = classification.strip()  
                
                self.verbose('\tprocessing article %s' % (article_id))
                
                #learn classification labels
                if classification not in self.categories:
                    self.categories.append(classification.strip()) 
                    
                class_count[classification]+=1
    
                article_path = os.path.join(training_dir,article_id+'.txt')
                content = self.parse_article(article_path)  
                for word in content:
                    if word not in self.keywords:
                        self.keywords.append(word) #new feature
#                    self.key_count[word]+=1
#                    self.key_class[word][classification]+=1
                    
                self.training_articles.append(Article(article_id,classification,content))
        
        for category in self.categories:
            self.prior[category] = class_count[category]/float(len(self.training_articles))
            self.verbose('\tprior(%s): %r' % (category, self.prior[category]))

             
        self.verbose('vectors successfully initialized')
    
    def train(self, categories, training_articles, keywords, key_class, class_count): 
            '''Train a naive Bayes model using add-one smoothing'''
            
            n = len(training_articles) #number of training documents
            prior = {} #p(C) -> prior probability (weight) of a class
            likilihood = defaultdict(dd0) #p(w|C) = w/sum(w) -> likelihood (weight) of a word given a class
            
            self.verbose('beginning model training')
            for category in categories:
                prior[category] =  class_count[category]/float(n) 
                self.verbose('\tprior(%s): %r' % (category, prior[category]))
                #all words in class
                words_in_category = []
                #all articles in class
                articles_of_category = [article for article in training_articles if article.classification == category]    
                for article in articles_of_category:
                    words_in_category.extend(article.content)
                    
                self.verbose('\twords in category %s: %i \n' % (category,len(words_in_category)))
                
                '''calculate conditional probability of word given class:
                adding one to m keywords = m. total words in category = |words in category|'''   
                total = len(words_in_category) + len(keywords)
                for word in keywords:
                    #add one -> minimum key_class value of one
                    likilihood[word][category] = (key_class[word][category]+1)/float(total)  
            
            #classification model
            model = prior, likilihood
            self.verbose('model training successful')
            return model
        
    def test(self, model, article):
        '''Apply model to test article'''
       
        prior, likilihood = model
        most_probable = None
        highest_probability = 0.0
        
        self.verbose('article: %s' % article.id)
        #probability of class given article = posterior(c|a)
        for category in self.prior.iterkeys():
            #log_b(x*y) = log_b (x) + log_b (y) -> avoid infinitesimal small values 
            posterior = math.log(prior[category] + 1)
            #adding 1 to avoid negative values for log(x<1)
            for word in article.content:
                posterior += math.log(likilihood[word][category] + 1)
            
            self.verbose("p(%s|%s) = %r" % (category, article.id, posterior))
            if (posterior > highest_probability):
                highest_probability = posterior
                most_probable = category
        
        return most_probable
    
    def train_models(self, n):
        '''Train N naive Bayes models'''
        
        articles = set(self.training_articles)
        traing_sample_size = len(articles)/n
        for _ in xrange(n):
            #pick a random subset of articles on which to train model
            training_set = random.sample(articles,traing_sample_size)
            #training sets are mutually exclusive, therefore subtract random selection
            articles = articles - set(training_set)
            
            keywords = []
            categories = []
            class_count = defaultdict(give_me_0) #count occurrences of each category
            #key_count = defaultdict(give_me_0) #keyword frequency count
            key_class = defaultdict(dd0)# keyword given class frequency count [][]
            
            for article in training_set:
                classification = article.classification
                class_count[classification]+=1
                
                if classification not in categories:
                    categories.append(classification) 
                    
                for word in article.content:
                    if word not in keywords:
                        keywords.append(word) #new feature
                    # key_count[word]+=1
                    key_class[word][classification]+=1
            
            model = self.train(categories, training_set, keywords, key_class, class_count)
            self.models.append(model)
        
        return self.models
    
    def run_tests(self, testing_dir, testing_file):   
        '''Construct the set of test articles, perform evaluation, and re '''
        
        testing_articles = [] #the set of training articles
        self.predictions = [] #set of test predictions
        overall_correct = 0 #overall correct
        overall_accuracy = 0 #overall accuracy
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
        #perform actual tests
        for article in testing_articles:
            vote = defaultdict(give_me_0)
            popular_vote = 0
            prediction = None
            for model in self.models:
                hypothesis = self.test(model, article)
                #each classifier gets 1 vote, but that vote is weighted by the prior probability of the label
#                vote[hypothesis] += self.prior[hypothesis]
                vote[hypothesis] += 1
                if vote[hypothesis] > popular_vote:
                    prediction = hypothesis
                    popular_vote = vote[hypothesis]
            
            #prediction for article has already been determined at this point
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

    def preprocess(self, text=''):
        '''
        Given some text, return a list of words that correspond to desired parts of speech
        Only want to consider: nouns, proper nouns,and verbs 
        (presentense, pastence verb, present participle, past participle
        '''
        
        #return only the words in the line
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
                for prediction in self.predictions:
                    of.write('%s: %s, %s\n' % (prediction[0], prediction[1], prediction[2]))
        except Exception, e:
            print 'error writing predictions to file: ', e
        
    def verbose(self, message = ''):
        '''Print messages only if flag is true'''
        if self.verbose_mode:
            print message


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
    
    data_path = 'ensemble_data.p'
    model_path = 'ensemble_model.p'
    
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
            print 'no serialized data found'
            nbac = NaiveBayesArticleClassifier(training_dir,training_file,verbose_mode)
            with open(data_path, 'wb') as data_pickle:
                pickle.dump(nbac, data_pickle)
            print 'data successfully serialized'
        except Exception,e:
            #failing to pickle doesn't halt execution (naturally)
            print 'error computing/pickling data:' , e
            
    if os.path.isfile(model_path) and not clean:
        try:
            #why? may want to use same training model on different testing data
            print 'found existing model'
            with open(model_path, 'rb') as model_pickle:
                nbac.models = pickle.load(model_pickle)
            print 'model loaded'
        except Exception, e:
            print 'error unpickling model: ', e
    else:
        try:
            print 'no serialized model found'
            models = nbac.train_models(3)
            with open(model_path, 'wb') as model_pickle:
                pickle.dump(models, model_pickle)
            print 'model successfully serialized'
        except Exception,e:
            print 'error computing/pickling model:' , e
     
    print 'beginning testing phase'
    acc, class_acc = nbac.run_tests(testing_dir, testing_file)
    
    print 'writing predictions to file'
    nbac.write_predictions(out_file)
    print 'successfully wrote predictions to file'
    
    print 'overall accuracy: ', acc
    for c in nbac.categories:
        print 'accuracy[%s]: %r' % (c, class_acc[c])
