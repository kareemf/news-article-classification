'''
Created on Apr 19, 2012

@author: reem
'''

class Article(object):
    '''
    Data representation of a news article
    '''
    
    def __init__(self,id=None,classification=None,contents=''):
        '''
        Constructor
        '''
        self.id = id
        self.classification = classification
        self.content = contents