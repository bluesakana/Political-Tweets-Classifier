def get_query(username, since=None, until=None):
    '''
    Function to create a query (str) to use with snscrape to scrape Twitter data
    
    Parameters
    ----------
    username : str
        twitter handle of account to be scraped
    
    since : str
        date from which the start scrape in format YYYY-MM-DD
        default = None
    
    until : str
        date till which the start scrape in format YYYY-MM-DD
        default = None
        
    Returns
    -------
    str
        in the format of "from:<username> since:<since> until:<until>"
        if <since> and <until> are not null input
        
    '''
    query = ''
    
    # append username
    query += f'from:{username}'
    
    # check whether until and since arguments are not null, then append
    if not until == None:
        query += f' until:{until}'
    
    if not since == None:
        query += f' since:{since}'

    return query


def scrape_tweets(username, since=None, until=None, attributes=None):
    '''
    Function to create a query (str) to use with snscrape to scrape Twitter data
    
    Parameters
    ----------
    username : str
        twitter handle of account to be scraped
        
    since : str
        date from which the start scrape in format YYYY-MM-DD
        default = None
    
    until : str
        date till which the start scrape in format YYYY-MM-DD
        default = None

    attributes : list
        list of attributes of the tweet to be scrapped, list should comprise of str
        default = None, all attributes will be scraped
        
    Returns
    -------
    pandas dataframe
        columns = attributes
        rows = tweets scraped
    
    '''
    
    import snscrape.modules.twitter as sntwitter
    import pandas as pd
    
    # get the query string required
    q = get_query(username, since, until)
    
    # create empty list of tweets attributes
    tweets = []
    
    # iterate through each tweet and scrape the relevant attributes
    print(f'Starting scrape of Twitter account {username}...')

    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(q).get_items()):
        
        # append all attributes to tweets list        
        tweets.append(list(vars(tweet).values()))
        
        # for every 100 tweets scraped, print progress on console
        if (i % 100 == 0) & (i >= 100):
            print(f'...Scraped {i} tweets...', end='\r')
            
    print(f'Scraped a total of {i} tweets.')

    # convert list into dataframe for easier manipulation
    tweets = pd.DataFrame(tweets, columns=list(vars(tweet).keys()))
    
    # flag to check that the while loop should be exited    
    end = False
    while(not end):
        # if attributes is None, return all the attributes
        if attributes == None:
            break
        # else, extract only indicated attributes and exit the while loop
        try:
            tweets = tweets.loc[:,attributes]
            end = True
        # if one or more of the indicated attributes is not in the list of attributes scraped, raise error
        except KeyError as err:
            print(err)
            # ask user for input of revised indicated attributes
            # if user gives blank input, extract all attributes instead
            attributes = input(prompt='You have entered the attributes list {}\n'.format(attributes) +
                               'Please input an alternative list of attributes in a list format\n' +
                               '(leave blank if you wish to extract all available attributes instead):') or None
            if not attributes == None: attributes = eval(attributes)
    
    return tweets


def compute_content_length(row, content_column, date_column, date_char_limit):
    '''
    Function to compute character length of tweet as a proportion of the maximum allowable length
    
    Parameters
    ----------
    row : pandas series object
        a row of tweet characteristics to be used, it should contain a column specified by the variable content_column and a date column specified by the variable date_column
        
    content_column : str
        name of the column that contains the tweet content
    
    date_column : str
        name of the column that contains the date of the tweet

    date_chat_limit : datetime
        datetime object that is the date at which the maximum tweet length was increased from 140 to 280
        
    Returns
    -------
    float
        the proportion of maximum tweet length 
    
    '''
    length = len(row[content_column])
    if row[date_column] < date_char_limit:
        return length / 140
    else:
        return length / 280
    

def compute_content_length(row, content_column, date_column, date_char_limit):
    '''
    Function to compute character length of tweet as a proportion of the maximum allowable length
    
    Parameters
    ----------
    row : pandas series object
        a row of tweet characteristics to be used, it should contain a column specified by the variable content_column and a date column specified by the variable date_column
        
    content_column : str
        name of the column that contains the tweet content
    
    date_column : str
        name of the column that contains the date of the tweet

    date_chat_limit : datetime
        datetime object that is the date at which the maximum tweet length was increased from 140 to 280
        
    Returns
    -------
    float
        the proportion of maximum tweet length 
    
    '''
    length = len(row[content_column])
    if row[date_column] < date_char_limit:
        return length / 140
    else:
        return length / 280