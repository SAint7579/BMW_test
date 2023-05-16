# BMW Challenge: Query to Boolean Conversion

### Description: 
The application takes in queries from users in a natural language format and converts them into a request body with a boolean code. The program has been written in Python 3.9 and the following are the main libraries used:
- Spacy
- TextBlob
- Streamlit
- DateFinder

### Walkthrough

### Description of the Code:

- <b> Utilities/utilities.py </b> : This contains basic functions to compare text. The main function used from this is the lcs_similaity that calculates the longest common subsequence similarity between two texts. (There are also some experimental functions using BERT to compare texts but they haven't been utilized).
- <b> Utilities/pos_tagging.py </b> : These contains functions to find the key elements such and their features and accumulate them in form of tags. These are tags are then used to find the model code and the boolean logic. 
- <b> Utilities/request_body_creation.py </b> : These are the main functions used for creating the request body. It contains methods to segregate tags into the 4 given categories, get the connotation of a tag, find the boolean logic and the model type and combine them into the request body format.
- <b> Method_test.py </b>: It contains code to test the basic functions and run the test cases from `test_cases.csv`. It also contains markdown text to show the process of building this application.
- <b> interface.py </b>: It contains the streamlit code for the user interface. 

### How to run the code:
- You can either install the requirements `pip install -r requirements.txt` or create a virtual environment from the `Env/environment.yml` file.
- Run the file `run.py` and the application will run on your local host. (Note: On running the code, the `en_core_web_md` model and the NLTK corpora will be downloaded if they are not available in the environment alreay).
- If there are any issues with the dependencies, the streamlit application is hosted on https://saint7579-bmw-test-interface-7cvopu.streamlit.app/.


### 
