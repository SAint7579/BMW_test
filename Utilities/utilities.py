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


def lcs_similarity(s1, s2):
    """
    Calculate the longest common subsequence (LCS) similarity between two strings. This is the standard 
    Dynamic Programming method to calculate LCS.
    
    Parameters
    ----------
    s1 : str
        First string.
    s2 : str
        Second string.
    
    Returns
    -------
    float
        LCS similarity between two strings.
    
    """
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
    # *Important* - Normalizing by the minimum length of the two strings to condition on the prompt
    # If we divide by max, then iX gives a better score for 318i than the iX models
    return L[m][n] / min(m, n)