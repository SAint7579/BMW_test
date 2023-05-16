import streamlit.web.cli as stcli
import os, sys
import spacy

def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path


if __name__ == "__main__":

    sys.argv = [
        "streamlit",
        "run",
        resolve_path("interface.py"),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())