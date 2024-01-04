import streamlit as st
import requests

SERVER_URL = 'https://linear-model-service-model-yoloogroth.cloud.okteto.net/v1/models/map-model:predict'

def make_prediction(inputs):
    predict_request = {'instances': inputs}
    response = requests.post(SERVER_URL, json=predict_request)
    
    if response.status_code == 200:
        prediction = response.json()
        return prediction
    else:
        st.error("Failed to get predictions. Please check your inputs and try again.")
        return None

def main():
    st.title('Predictor de ubicaciones geográficas')

    st.header('Coordenadas para Alemania')
    alemania_lat = st.number_input('Ingrese la latitud de Alemania:', value=51.1657)
    alemania_lon = st.number_input('Ingrese la longitud de Alemania:', value=10.4515)

    st.header('Coordenadas para Bolivia')
    bolivia_lat = st.number_input('Ingrese la latitud de Bolivia:', value=-16.5000)
    bolivia_lon = st.number_input('Ingrese la longitud de Bolivia:', value=-64.6667)

    if st.button('Predecir'):
        inputs = [
            [alemania_lon, alemania_lat],
            [bolivia_lon, bolivia_lat]
        ]
        predictions = make_prediction(inputs)

        if predictions:
            st.write("\nPredicciones para Alemania:")
            st.write(predictions['predictions'][0])

            st.write("\nPredicciones para Bolivia:")
            st.write(predictions['predictions'][1])

if __name__ == '__main__':
    main()
