'''
Created on Apr 29, 2012

@author: Kareem
'''
import traceback
import os
import urllib2
import socket
from BeautifulSoup import BeautifulSoup

def scrape_news(articles_per_category = 1):
    #Google News scraper
    google_categories  = {'Sports' : 's', 'Entertainment': 'e', 'Health' : 'm', 'Business' : 'b', 'Sci/Tech' : 'tc'}
    
    #TODO: take as arg
    articles = []
    
    timeout = 60
    socket.setdefaulttimeout(timeout)
    
    for c in google_categories.keys():
        print c + ':'
        
        #Google News aggregate news list by topic
        url = 'http://news.google.com/news/section?pz=1&cf=all&ned=us&topic=%s&ict=ln' % google_categories[c]
        page = urllib2.urlopen(url).read()
       
        #the html parser
        soup = BeautifulSoup(page.decode('utf8', 'ignore'), convertEntities=BeautifulSoup.HTML_ENTITIES) #decoding before BS gets to work saves lives 
        
        #properties to look for
        title_soup = soup.findAll('span', 'titletext') 
        link_soup  = soup.findAll('a', 'article') 
        
        article_id = 0
        
        for i  in xrange(articles_per_category):
            try:  #avoid index out of bounds
                article_title = title_soup[i].getText()
                article_url = link_soup[i]['href']
                
                print article_title
                
                try:
                    #open article's source link
                    article_page = urllib2.urlopen(article_url).read()
                    article_soup = BeautifulSoup(article_page, convertEntities=BeautifulSoup.HTML_ENTITIES).findAll('p')
                    
                    #first line of outpur is going to be the title
                    article = article_title + '\n' 
                    for s in article_soup:
                        article += s.getText() + '\n'
#                    articles.append([article, google_categories[c]])
                    article_id = write_out(article, article_id, google_categories[c])
                    
                except UnicodeEncodeError:
                    #can't access article, skip it
                    print 'WARNNING: RAN INTO ENCODING TROUBLE'
                    continue
                except urllib2.HTTPError:
                    #can't access article, skip it
                    print 'WARNNING: RAN INTO HTTP TROUBLE'
                    continue    
                except urllib2.URLError:
                    #can't access article, skip it
                    #probably becaause they don't like bots...
                    print 'WARNNING: RAN INTO URL TROUBLE'
                    continue    
                except:
                    print 'UNKNOWN ERROR'
                    continue
            except IndexError, e:
                print 'No more articles in category'
                break
        print '\n'
    
    return articles

def write_out(article, article_id, category, directory = '', project_categories = ['s', 'b', 't', 'h' ,'e']):
    if directory == '':
        current_dir = os.curdir
        directory = os.path.join('.'+current_dir,'training_articles')
        
    if len(article) > 200:
        try:
            with open(os.path.join(directory, article_id + '.txt'), 'w') as out:
                out.write(article)
            with open(os.path.join(directory, 'class.txt'), 'a') as out: 
                if category not in ['s', 'b', 't', 'h' ,'e']:
                    category = 'h' if category is 'm' else 't' 
                out.write(article_id + ', ' + category + '\n')
            return article_id+1
        except UnicodeEncodeError:
            #can't access article, skip it
            print 'Failed to write article %s because of UnicodeEncodeError' % article_id
            continue
        except:
            traceback.print_exc()
            continue
    else:
        print 'Article is excessively short, discarding'
        return article_id
                 
    
def batch_write_file(articles):       
    '''write all articles to one file'''
    with open('out.txt','w') as out:
        for article, category in articles:
            #excessively short articles usually stem from errors in parsing and wouldn't improve trainng results
            if len(article) > 200:
                try:
                    out.write(category+ '\n' + article + '\n\n')
                except UnicodeEncodeError:
                    #can't access article, skip it
                    traceback.print_exc()
                    continue 
            else:
                print 'Article is excessively short, discarding' 
                   
def batch_write_folder(directory, articles, project_categories): 
    '''write each article to separate, numbered file + update class.txt'''
    for i in xrange(len(articles)):
        article, category = articles[i]
        if len(article) > 200:
            try:
                article_id = str(i+1)
                with open(os.path.join(directory,article_id + '.txt'), 'w') as out:
                    out.write(article)
                with open(os.path.join(directory,'class.txt'), 'a') as out: 
                    if category not in project_categories:
                        category = 'h' if category is 'm' else 't'
                        
                    out.write(article_id + ', ' + category + '\n')
            except UnicodeEncodeError:
                #can't access article, skip it
                print 'Failed to write article %s because of UnicodeEncodeError' % article_id
                continue
            except:
                traceback.print_exc()
                continue
        else:
            print 'Article is excessively short, discarding'
            
#TODO: Introduce date variable, allow for pulling articles for random points in time
#if articles in the same category are from roughly the same peroid of time, there is a high chance that they will be about the same thing
#This may be detremental to training by causing low varitations in topics with a given classification.
#For example, sports articles from the same week will have a high likelyhood of being about the same sport and maybe even same specific event, but 
#a testing article from a month later may be about a totally differnt sport, and the model will perform poorly against the newly introduced terms.
if __name__ == '__main__':
    current_dir = os.curdir
    training_dir = os.path.join('.'+current_dir,'training_articles')
    
    articles = scrape_news(100)
    print 'Done with crawling/scraping'#+' writing articles to folder'
    #batch_write_file(training_dir, articles, ['s', 'b', 't', 'h' ,'e'])              
    #print 'Done with file output.'
