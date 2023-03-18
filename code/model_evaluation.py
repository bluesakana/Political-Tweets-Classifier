def fill_na_with_blank(df):
    '''
    Custom function to fill missing values with '' string
    This will be used as a FunctionTransformer in the pipeline
    
    Paramters:
    ----------
    df: pandas Series
        column of data to be filled with '' strings
        
    Returns:
    --------
    pandas Series
        Series with missing values replaced with '' string
    
    '''
    import pandas as pd
    
    return df.fillna('')

def lemmatize_column(df):
    '''
    Custom function to lemmatize strings in a pandas Series
    
    Paramters:
    ----------
    df: pandas Series
        column of data with strings to be lemmatized
    
    Returns:
    --------
    pandas Series
        Series of lemmatized words in a string separated by ' '
    
    '''
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    
    import pandas as pd
    
    lemmatizer = WordNetLemmatizer()
    lemm_col = []

    for text in df:
        tokens = word_tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        lemm_col.append(' '.join(lemmatized_tokens))

    return pd.Series(lemm_col, name=df.name)


def avg_f1_score(y_true, y_pred):
    '''
    Function that computes the average F1 score for a binary classifier
    
    Parameters:
    -----------
    y_true: pandas Series
        list of actual labels
    
    y_pred: pandas Series
        list of predicted labels
        
    Returns:
    --------
    float
        average F1 score
        
    '''
    
    from sklearn.metrics import f1_score
    
    f1_0 = f1_score(y_true, y_pred, pos_label=0)
    f1_1 = f1_score(y_true, y_pred, pos_label=1)
    return (f1_0 + f1_1) / 2

def compute_score(model, X, y):
    '''
    Function that extracts the avg f1 score from the model and computes the avg f1 score based on the X and y input
    
    Parameters:
    -----------
    model: sklearn RandomizedSearchCV class
        sklearn object with the predict method and best_score_ attribute
        
    X: pandas dataframe
        predictor variables
        
    y: pandas Series
        outcome variables (true values)
        
    Returns:
    --------
    score_1 float:
        the score computed using X and y, applied to the model
        
    score_2 float:
        the cross-val score as computed by the model
        
    '''
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import pprint
    
    # compute scores
    score_1 = avg_f1_score(y, model.predict(X))
    score_2 = model.best_score_
    
    # output scores on screen
    print(f'Train Average F1 Score: \t{score_1:.3f}')
    print(f'{model.cv}-Fold CV Average F1 Score: \t{score_2:.3f}')
    
    # plot confusion matrix
    cm_plot = ConfusionMatrixDisplay(confusion_matrix(y, model.predict(X)), display_labels=['wp', 'pap'])

    cm_plot.plot(cmap='Blues')
    
    # print optimal hyperparameters
    pprint.pprint(model.best_params_)

    return score_1, score_2


def compute_score_specific(model, X, y):
    '''
    Function that computes the score and confusion matrix of a specific model
    
    Parameters:
    -----------
    model: sklearn class
        sklearn object with the predict method
        
    X: pandas dataframe
        predictor variables
        
    y: pandas Series
        outcome variables (true values)
        
    Returns:
    --------
    score_1 float:
        the score computed using X and y, applied to the model
        
    '''
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import pprint
    
    # compute scores
    score_1 = avg_f1_score(y, model.predict(X))
    
    # output scores on screen
    print(f'Average F1 Score: \t{score_1:.3f}')
    
    # plot confusion matrix
    cm_plot = ConfusionMatrixDisplay(confusion_matrix(y, model.predict(X)), display_labels=['wp', 'pap'])

    cm_plot.plot(cmap='Blues')

    return score_1


def get_wrongs(model, X_train, y_train):
    '''
    Function that pulls out the wrong predictions
    
    '''
    
    y_preds = model.predict(X_train)
    
    temp = X_train.copy()
    temp['proba'] = model.predict_proba(X_train)[:,1]
    
    pap_wrong = temp.loc[(y_train==1) & (y_preds==0),:].sort_values(by='proba', ascending=True)
    
    wp_wrong = temp.loc[(y_train==0) & (y_preds==1),:].sort_values(by='proba', ascending=False)
    
    return pap_wrong, wp_wrong


def process_coef(df):
    '''
    Function that processes the raw coefficients from Logistic Regression model into groups and aspects for visualisation
    
    Parameters:
    -----------
    df: pandas Dataframe
        Dataframe of coefficients, with the index being the feature names
        
    Returns:
    --------
    pandas Dataframe
        Dataframe with the index being the full feature names and the coefficient column is preserved, a column called 'variable' which is the feature names, a column called 'group' which is the preprocessor transformer names, and a column called 'aspect' which contains the 4 aspects of tweets
        
    '''
    
    df['variable'] = df.index.str.split('__').str.get(-1).tolist()
    df['group'] = df.index.str.split('__').str.get(0).tolist()
    
    df['aspect'] = ''
    
    df.loc[
        (df['group'] == 'content_transformer') |
        (df['group'] == 'hashtag_transformer') |
        (df['variable'] == 'content_length'),
        'aspect'
    ] = 'messaging'
    
    df.loc[
        (df['group'] == 'remainder') & 
        (df['variable'] == 'sentiment_score'),
        'aspect'
    ] = 'tone'
    
    df.loc[
        (df['group'] == 'remainder') & 
        (df['variable'].str.contains('num_media_')),
        'aspect'
    ] = 'content_type'

    df.loc[
        (df['group'] == 'remainder') & 
        (df['variable'].str.contains('num_links_')),
        'aspect'
    ] = 'content_type'
    
    df.loc[
        (df['group'] == 'remainder') & 
        (df['variable'] == 'part_of_convo'),
        'aspect'
    ] = 'engagement_level'

    df.loc[
        (df['group'] == 'remainder') & 
        (df['variable'].str.contains('Count')),
        'aspect'
    ] = 'engagement_level'
    
    df.sort_values(by='coefficient', ascending=False, inplace=True)
    
    return df