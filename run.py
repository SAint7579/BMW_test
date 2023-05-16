import streamlit.web.cli as stcli
import os, sys
import spacy

def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path


if __name__ == "__main__":

    ## Installing some models required by spacy if it is not already installed
    try:
        spacy.load("en_core_web_md")
    except OSError:
        os.system("python -m spacy download en_core_web_md")


    sys.argv = [
        "streamlit",
        "run",
        resolve_path("interface.py"),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())