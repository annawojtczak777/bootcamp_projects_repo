# https://365datascience.com/tutorials/machine-learning-tutorials/how-to-deploy-machine-learning-models-with-python-and-streamlit/

import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict
from model import  exploration, estimation_models, predict_test, krzywa_ROC
import time

st.set_page_config(page_title='Asteroid Hazard Prediction Model')
st.title('Asteroid Hazard Prediction Model')

st.image(
        "https://cdn.pixabay.com/photo/2019/07/29/20/51/asteroid-4371514_960_720.jpg",
        use_column_width='auto'
        )

st.header('Intro')

st.markdown('''Near Earth Objects (NEOs) are asteroids and comets which have orbits passing near the Earth.
They range in size from less than one to tens of kilometres across. 

There are over 1.1 million asteroids known in our Solar System, with over 30,000 of these being NEOs.
Despite the large number of them, the chance of a NEO striking the Earth is slim.

Although rare, NEOs are still a future threat,and the consequences of a bigger object hitting us could potentially be catastrophic for life on Earth.
Keeping track of Near Earth Objects and the likelihood of them making impact with Earth is therefore very important.''')

st.write('---')

st.subheader('Dataset')
st.markdown('The dataset compiles the list of NASA certified asteroids that are classified as the nearest earth object.')
df = pd.read_csv("neo_v2.csv")
st.write(df.head())

exploration()

estimation_models()

predict_test()

krzywa_ROC()


st.write('---')
st.header('Try our model and check hazard of your Asteroid!')

st.subheader('Choose the variables describing the Asteroid')
col1, col2= st.columns(2)
col3, col4 = st.columns(2)

with col1:
    st.text('Estimated_Diameter_Max')
    est_diameter_max = st.slider('Maximum Estimated Diameter in Kilometres', 0.001362 , 84.730541)

with col2:
    st.text('Relative_Velocity')
    relative_velocity = st.slider('Velocity Relative to Earth in km/h', 203.346433, 236990.128088)

with col3:
    st.text('Miss_Distance')
    miss_distance = st.slider('Distance in Kilometres missed', 6.75, 74.8) #to jest do poprawki

with col4:
    st.text('Absolute_Magnitude')
    absolute_magnitude = st.slider('Describes intrinsic luminosity', 9.230000, 33.200000)


if st.button('Predict type of Hazordous'):
    start_time = time.time()
    result = predict(np.array([[est_diameter_max, relative_velocity, miss_distance, absolute_magnitude]]))
    elapsed_time = time.time() - start_time
    if result[0] == 1:
        st.subheader('The asteroid is hazardous.')
        st.text(f'Elapsed time to compute the predict: {elapsed_time:.3f} seconds')
    else:
        st.subheader('The asteroid is not hazardous.')
        st.text(f'Elapsed time to compute the predict: {elapsed_time:.3f} seconds')
