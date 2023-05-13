from utilities import lcs_similarity, softmax
from pos_tagging import *

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
    """Segregates the tags into 4 lists: Model Type, Steering Config, Package, Roof Config

    Args:
        tags (list): List of dictionaries containing the tags and their POS tags

    Returns:
        list: List of lists containing the segregated tags
    """
    # Four Lists: Model Type, Steering Config, Package, Roof Config

    segregated = [[]]*4

    for t in tags:
        tag_text = t['text']

        # Getting the match score with each dictionary
        match_score = [ max([lcs_similarity(tag_text,i,type='min') for i in list(MODEL_TYPE_CODE.keys())]),
                        max([lcs_similarity(tag_text,i,type='min') for i in list(STEERING_CONFIG_CODE.keys())]),
                        max([lcs_similarity(tag_text,i,type='min') for i in list(PACKAGE_CODE.keys())]),
                        max([lcs_similarity(tag_text,i,type='min') for i in list(ROOF_CONFIG_CODE.keys())])]
        

        if max(match_score) >= 0.5:
            segregated[np.argmax(match_score)] = segregated[np.argmax(match_score)] + [t]

    return segregated

