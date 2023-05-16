
import spacy
from utilities import lcs_similarity
import os

## Installing some models required by spacy if it is not already installed
if "en_core_web_md" not in spacy.util.get_installed_models():
    # Download the model
    spacy.cli.download("en_core_web_md")

nlp = spacy.load("en_core_web_md")




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
        if child.dep_ in ['compound', 'amod', 'nummod']:
            if len(list(child.children)) > 0:
                # Call recursively if the child have more children
                recursive_compound_extraction(child, compounds)
            compounds.append(child)

    # Return the compounds
    return compounds


def get_key_terms_with_pos(text):
    """
    Extracts key terms from a text and returns them with their POS tags.
    
    Parameters
    ----------
    text : str
        Input text.
    
    Returns
    -------
    list of dict
        List of dictionaries containing the key terms and other important pos tag information.
    
    """
    global nlp
    doc = nlp(text)

    all_tags = []
    for token in doc:
        # The 2nd part of the if condition is because if there is any additional noun around these terms, they are not recognized by Spacy
        # A .lower() on the entire text does solve this, but it messes up the POS for the other terms (the roof configurations mainly)
        if (token.pos_  in ["NOUN", "PROPN", "PRON", "X"]) or (token.text.lower() in ['xdrive50', 'xdrive40', 'xdrive40i','xdrive40d', 'ix', 'x7']):

            if token.pos_ == "PRON":
                # This is mainly for the 'iX' type keywords that are recongized as pronouns.
                # We want to make sure that these are compound words
                if len(list(token.children)) == 0:
                    continue

            # Getting the full term
            compound = recursive_compound_extraction(token, [])
            compound.append(token)

            # HACK: This is to deal with the problem of xDrive40 (specifically) not being recongized as a compound word
            if len(compound) == 1:
                # Checking if the word is similar to xDrive with lcs
                if lcs_similarity(compound[0].text, 'xDrive', type='min') > 0.8:
                    # include the previous word from the doc
                    compound = [doc[token.i-1]] + compound
                    # This is not going to harm the LCS score even if the previous word is not iX or X7
            
            tags = {'values': compound,
                    'main_token': token,
                    'text': ' '.join([i.text for i in compound]),
                    'child_conj': [i for i in token.children if i.pos_ == 'CCONJ'],
                    'head_conj': [i for i in token.head.children if i.pos_ == 'CCONJ'],
                    'pos': token.pos_,
                    'dep': token.dep_,
                    'head': token.head,}
            
            all_tags.append(tags)

    # Reduction Logic: This is because just adding NOUNS is creating redndant tags
    all_values = [i['values'] for i in all_tags]

    unique_tags = []
    for t in all_tags:
        # Logic: If there are more than one tag that contains all the values of the current tag, then there is a superset persent in all_tags
        number_of_supersets = sum([len(set(t['values'])-set(all_tags[0])-set(j)) == 0 for j in all_values])

        # There should only be one superset (itself)
        if number_of_supersets==1:
            unique_tags.append(t)

    # Returning all_tags just in case we need some head not captured in the unique_tags
    return unique_tags, all_tags