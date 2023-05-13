
import spacy
nlp = spacy.load("en_core_web_sm")

def recursive_compound_extraction(token, compounds = []):
    """
    Recursively extract compound and amod children of a token.
    
    Parameters
    ----------
    token : spacy.tokens.token.Token
        Input token.
    compounds : list of spacy.tokens.token.Token
        List of compound words.
    
    Returns
    -------
    list of spacy.tokens.token.Token
        List of compound words.
    
    """
    # Go through all children of the token
    for child in token.children:
        if child.dep_ == 'compound' or child.dep_ == 'amod':
            if len(list(child.children)) > 0:
                # Call recursively if the child have more children
                recursive_compound_extraction(child, compounds)
            compounds.append(child)

    # Return the compounds
    return compounds

