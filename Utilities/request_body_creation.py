from utilities import lcs_similarity, softmax
from pos_tagging import *
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import numpy as np

MODEL_TYPE_CODE = { 'iX xDrive50': '21CF',
                    'iX xDrive40': '11CF',
                    'X7 xDrive40i': '21EM',
                    'X7 xDrive40d': '21EN',
                    'M8': 'DZ01',
                    '318i': '28FF',}

STEERING_CONFIG_CODE = { 'Left-Hand Drive': 'LL',
                         'Right-Hand Drive': 'RL',}

PACKAGE_CODE = {    'M Sport Package': 'P337A',
                    'M Sport Package Pro': 'P33BA',
                    'Comfort Package EU': 'P7LGA'}

ROOF_CONFIG_CODE = {'Panorama Glass Roof': 'S402A',
                    'Panorama Glass Roof Sky Lounge': 'S407A',
                    'Sunroof': 'S403A'}


def segregated_tags(tags):
    """
    Segregate the tags into different types.
    
    Parameters
    ----------
    tags : list
        List of tags.
    
    Returns
    -------
    list
        List of the category number of each tag.
    
    """
    # Adding the types to the tags
    
    segregated = []

    for t in tags:
        tag_text = t['text']

        # Getting the match score with each dictionary
        match_score = [ max([lcs_similarity(tag_text,i,type='min') for i in list(MODEL_TYPE_CODE.keys())]),
                        max([lcs_similarity(tag_text,i,type='min') for i in list(STEERING_CONFIG_CODE.keys())]),
                        max([lcs_similarity(tag_text,i,type='min') for i in list(PACKAGE_CODE.keys())]),
                        max([lcs_similarity(tag_text,i,type='min') for i in list(ROOF_CONFIG_CODE.keys())])]
        
        
        if max(match_score) >= 0.5:
            segregated.append(np.argmax(match_score))
        else:
            segregated.append(-1)

    return segregated
 

def revert_tense(word, tense):
    """
    Reverts the tense of a word. This is needed because simple lemmatisation did not work.
    
    Parameters
    ----------
    word : str
        Input word.
    tense : str
        Tense to revert to.
    
    Returns
    -------
    str
        Reverted word.
    
    """
    blob = TextBlob(word)
    return blob.words[0].lemmatize(tense)

def get_word_sentiment(word):
    """
    Get the sentiment of a word.
    
    Parameters
    ----------
    word : str
        Input word.
    
    Returns
    -------
    float
        Sentiment of the word.
    
    """
    # Create a custom sentiment lexicon
    custom_lexicon = {'with': 0.5, 'include': 0.5, 'want': 0.5, 'have':0.5, 
                  'without': -0.5, 'exclude': -0.5}
    
    word = revert_tense(word, 'v')
    if word in custom_lexicon:
        return custom_lexicon[word]
    else:
        ## This is just in case a completely new word is encountered
        blob = TextBlob(word, analyzer=NaiveBayesAnalyzer())
        return blob.sentiment.p_pos - blob.sentiment.p_neg



## Rewrite get_boolean_logic_datastruct with numpydoc docstring format
def get_boolean_logic_datastruct(tags, segregated):
    """
    Get the boolean logic data structure from the tags and the segregated list.
    
    Parameters
    ----------
    tags : list
        List of tags.
    segregated : list
        List of the category number of each tag.
    
    Returns
    -------
    tuple
        Tuple containing the boolean logic data structure and the logic sentiment. This can later be merged into a string while making request body.
    
    """
    code_dictionaries = [STEERING_CONFIG_CODE, PACKAGE_CODE, ROOF_CONFIG_CODE]
    # Just so that we can look at the sentiment of the previous tag as well (if there is a conjugation)
    prev_tag = None
    prev_cat = None
    logic = []
    logic_sentiment = []
    for s,t in zip(segregated, tags):
        if s in [1,2,3]:
            print(t)

            ## Getting the sentiment of the head
            head_sentiment = get_word_sentiment(t['head'].text)

            if abs(head_sentiment) < 0.2:
                ## We use the previous one
                t['connotation'] = prev_tag['connotation'] if prev_tag else 'pos'
            else:
                # Checking if there is a negation in the children of head
                if 'neg' in [i.dep_ for i in t['head'].children]:
                    head_sentiment = -head_sentiment

                t['connotation'] = 'pos' if head_sentiment > 0 else 'neg'

            # Fetch the code dictionary
            code_dict = code_dictionaries[s-1]

            # Get the correct code from the dictionary
            code = None
            try:
                # First try to fetch with direct indexing
                code = code_dict[t['text']]
            except:
                # if not found, try to use the LCS similarity (Using max in this case)
                similarity_score = [lcs_similarity(t['text'],i,type='max') for i in list(code_dict.keys())]
                if max(similarity_score) >= 0.5:
                    code = list(code_dict.values())[np.argmax(similarity_score)]

            # If the code ends up being None, then the tag was misclassified

            ''' The crux of the logic:

            - The goal is to divide the boolean string into 2 parts: Sentiment and Code.
            - The code(s) will be joined with either / or + based on the conjugation of the previous term.
            - The code(s) will carry the - sign if the connotation is negative.
            - Since you can't ask for 'and' of multiple components of the same category, we are assuming 
            that they will be joined with a '/' only. codes that are to be put in parantheses would be kept in 
            the same list [].
            
            Example : "+(A/-B)+-C" will have the data structures: [[A,-B],[-C]] and signs: [+,+]
            
            '''
            if code is not None:
                if prev_tag is not None:
                    if prev_cat == s:
                        t['code'] = code if t['connotation']=='pos' else '-'+code
                        logic[-1].append(t)
                    else:
                        t['code'] = code if t['connotation']=='pos' else '-'+code
                        logic.append([t])
                        # The or conjugation between the two categories comes from the head of the fisrt tag of the previous group or the head of the current tag (if singular).
                        # Can be seen in the displacy image above
                        if (len(logic[-2][0]['head_conj']) > 0 and logic[-2][0]['head_conj'][0].text == 'or') or (len(t['head_conj']) > 0 and t['head_conj'][0].text == 'or'):
                            logic_sentiment.append('/')
                        else:
                            logic_sentiment.append('+')

                else:
                    t['code'] = code if t['connotation']=='pos' else '-'+code
                    logic.append([t])
                    logic_sentiment.append('+')

                # print([[i['code'] for i in j] for j in logic])
                # print(logic_sentiment)

            prev_tag = t
            prev_cat = s

    return logic, logic_sentiment