## Import libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


## Rewrite sbert embeddings with PEP257 docstring
def get_sbert_embeddings(text):
    """
    Get SBERT embeddings for input text.
    
    Parameters
    ----------
    text : str
        Input text.
    
    Returns
    -------
    numpy.ndarray
        SBERT embeddings for input text.
    
    """
    # Importing libraries
    import torch 
    from transformers import AutoTokenizer, AutoModel

    # Importing SBERT model and tokenizer
    model_name = 'sentence-transformers/stsb-distilbert-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize input text
    tokens = tokenizer.encode(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

    # Convert input to tensor and get SBERT embeddings
    outputs = model(tokens)
    sbert_embeddings = outputs[0]

    # Pooling strategy - Taking the mean embedding
    sbert_embeddings = sbert_embeddings.mean(dim=1)

    # Convert embeddings to NumPy array
    sbert_embeddings_np = sbert_embeddings.detach().numpy()

    return sbert_embeddings_np

def get_cosine_similarity(embedding1, embedding2):
    """
    Get cosine similarity between two embeddings.
    
    Parameters
    ----------
    embedding1 : numpy.ndarray
        First embedding.
    embedding2 : numpy.ndarray
        Second embedding.
    
    Returns
    -------
    float
        Cosine similarity between two embeddings.
    
    """
    # Calculate cosine similarity between two embeddings
    cosine_similarity_score = cosine_similarity(embedding1.reshape(1,-1), embedding2.reshape(1,-1))[0][0]

    return cosine_similarity_score


def softmax(x):
    """
    Apply softmax normalization to a numpy array of numbers.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input array of numbers.
    
    Returns
    -------
    numpy.ndarray
        Output array of numbers after softmax normalization.
    
    """
    # Apply softmax normalization to input array
    x = np.exp(x)
    x = x / np.sum(x)

    return x


def lcs_similarity(s1, s2, type='min', extreme_length=None):
    """
    Calculate the longest common subsequence (LCS) similarity between two strings. This is the standard 
    Dynamic Programming method to calculate LCS.
    
    Parameters
    ----------
    s1 : str
        First string.
    s2 : str
        Second string.
    type : str, optional
        Normalize with the min/max of the string length. The default is 'min'.
    extreme_length : int, optional
        The shortest/longest a string of this type can be. The default is None.
    
    Returns
    -------
    float
        LCS similarity between two strings.
    
    """
    # Convert to lowercase
    s1,s2 = s1.lower(), s2.lower()

    m, n = len(s1), len(s2)
    # Creating the matrix for storing match values
    L = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif s1[i-1] == s2[j-1]:
                # Increment if the characters match
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    # *Important* - Normalizing here would be different for different cases
    # If we divide by max, then iX gives a better score for 318i than the iX models
    # If we divide by min, there are problems with the overlapping names of other configuration.
    if type == 'max':
        if extreme_length:
            ## This is in case POS tagger adds additional words as compounds
            ## Essentially, the traget cannot be shorter than shortest key and longer than longest key
            din = min(max(m, n), extreme_length)
        else:
            din = max(m, n)

        return L[m][n] / din
    else:
        if extreme_length:
            din = max(min(m, n), extreme_length)
        else:
            din = min(m, n)
        return L[m][n] / din
    