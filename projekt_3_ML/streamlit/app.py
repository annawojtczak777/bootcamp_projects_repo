#https://365datascience.com/tutorials/machine-learning-tutorials/how-to-deploy-machine-learning-models-with-python-and-streamlit/
import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Wykrywanie Asteroid potencjalnie zagrażających Ziemi')
st.markdown('NASA - Nearest Earth Objects')
st.markdown('Aplikacja klasyfikacyjna, która przewiduje, czy w zalezności od podanych cech asteroida uderzy w Ziemię') 

st.header('Zmienne opisujące asteroidę')
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.text('est_diameter_max')
    est_diameter_max = st.slider('Max Estimated Diameter in Kilometres', 0.001362 , 84.730541)

with col2:
    st.text('relative_velocity')
    relative_velocity = st.slider('Velocity Relative to Earth', 203.346433, 236990.128088)

with col3:
    st.text('miss_distance')
    miss_distance = st.slider('Distance in Kilometres missed', 6.75, 74.8) #to jest do poprawki

with col4:
    st.text('absolute_magnitude')
    absolute_magnitude = st.slider('Describes intrinsic luminosity', 9.230000, 33.200000)

st.button('Predict type of Hazordous')

if st.button('Predict type of Hazordous'):
    result = predict(np.array([[est_diameter_max, relative_velocity, miss_distance, absolute_magnitude]]))
    st.text(result[0]) # to też wywala błąd
