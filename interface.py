import sys
sys.path.insert(1, 'Utilities')
from request_body_creation import *
import streamlit as st
import pandas as pd
import json


if __name__ == '__main__':

    ### This is the user Interface code build with streamlit.

    st.title("BMW Query Interface")

    # Adding the reference table just to cross check
    df1 = pd.DataFrame(MODEL_TYPE_CODE,index=['Code']).T
    df2 = pd.DataFrame(PACKAGE_CODE,index=['Code']).T
    df3 = pd.DataFrame(ROOF_CONFIG_CODE,index=['Code']).T
    df4 = pd.DataFrame(STEERING_CONFIG_CODE,index=['Code']).T

    upper_index = ['MODEL_TYPE_CODE'] * len(MODEL_TYPE_CODE) + ['PACKAGE_CODE'] * len(PACKAGE_CODE) + ['ROOF_CONFIG_CODE'] * len(ROOF_CONFIG_CODE) + ['STEERING_CONFIG_CODE'] * len(STEERING_CONFIG_CODE)
    dataframe = pd.concat([df1, df2, df3, df4], axis=0)
    index = [np.array(upper_index),np.array(dataframe.index)]
    html = pd.DataFrame(dataframe.values.reshape(-1), index=index, columns=['Code']).to_html()

    option = st.sidebar.selectbox('Select Option', ('Query','Reference Table'))

    if option == 'Reference Table':
        # Displaying the table in the main content area
        st.header('Reference Table')
        st.write(html, unsafe_allow_html=True)
    else:
        # Displaying the textbox in the main content area
        
        prompt = st.text_area("Query", value='If you want to add multiple query, separate them with ";"', height=200, max_chars=None, key=None)

        # Add a submit button
        submitted = st.button('Process')
        if submitted:
            propmts = prompt.split(';')

            request_bodies = []
            for i,p in enumerate(propmts):
                try:
                    request_bodies += get_request_body(p)[0]
                except Exception as e:
                    st.write(f'Error in query {i+1}: {e}')
                    continue

            if len(request_bodies) > 0:
                # Converting request bodies to json
                json_str = json.dumps(request_bodies, indent=1)

                # Putting the string in a text area
                st.text_area("Request Body", value=json_str, height=200, max_chars=None, key=None)

                submit_rb = st.button('Submit')

                if submit_rb:
                    # Add that to the database
                    pass




