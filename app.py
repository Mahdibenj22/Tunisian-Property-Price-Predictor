import pandas as pd
import joblib
import numpy as np
import shap
import streamlit as st
import time 


st.set_page_config(
    page_title=" Tunisian Property Price Predictor",
    page_icon="🏡",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stButton button {
        background-color: #2b5876;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #4e4376;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load the saved model
xgb_loaded_model = joblib.load('xgb_classification_model_updated.pkl')

# Define the mappings
category_mapping = {
    "Appartements": 0,
    "Locations de vacances": 1,
    "Magasins, Commerces et Locaux industriels": 2,
    "Maisons et Villas": 3,
    "Colocations": 4,
    "Bureaux et Plateaux": 5
}

region_mapping = {
    "La Soukra": 111, "Ariana Ville": 5, "Borj Louzir": 17, "Chotrana": 31, 
    "Jardins D'el Menzah": 87, "Raoued": 154, "Ennasr": 56, "Ghazela": 74,
    "Ariana": 4, "Sidi Thabet": 184, "Mnihla": 132, "Ettadhamen": 58,
    "Béja": 21, "Béja Nord": 22, "Testour": 201, "Téboursouk": 206,
    "Amdoun": 3, "Medjez el-Bab": 125, "Béja Sud": 23, "Medina Jedida": 124,
    "Fouchana": 63, "Mornag": 136, "Mégrine": 144, "El Mourouj": 50,
    "Hammam Lif": 80, "Radès": 153, "Mohamedia": 133, "Boumhel": 20,
    "Ben Arous": 9, "Hammam Chott": 78, "Ezzahra": 61, "Bizerte": 13,
    "Bizerte Nord": 14, "Ras Jebel": 155, "Mateur": 123, "Zarzouna": 211,
    "Ghar El Melh": 73, "Menzel Bourguiba": 127, "Bizerte Sud": 15,
    "Gabès Médina": 66, "Gabès Sud": 68, "Gabès": 65, "Métouia": 146,
    "Gabès Ouest": 67, "Mareth": 122, "Nouvelle Matmata": 149, 
    "El Hamma": 45, "Ghanouch": 72, "Gafsa Sud": 71, "Oum El Araies": 152,
    "Métlaoui": 145, "El Guettar": 44, "Gafsa": 69, "Gafsa Nord": 70,
    "Tabarka": 197, "Jendouba": 89, "Bou Salem": 18, "Ain Draham": 1,
    "Kairouan": 90, "Kairouan Sud": 92, "Kairouan Nord": 91, 
    "Sbikha": 171, "Haffouz": 77, "El Ouslatia": 54, "Chebika": 30,
    "Kasserine": 97, "Sbiba": 170, "Kasserine Nord": 98, "Sbeïtla": 169,
    "Fériana": 64, "Kasserine Sud": 99, "Oued Ellil": 150, 
    "Douar Hicher": 42, "Manouba Ville": 121, "La Manouba": 112, 
    "Mornaguia": 137, "Denden": 36, "Djedeida": 38, "Borj El Amri": 16,
    "Menzel El Habib": 130, "Tebourba": 200, "Kef Ouest": 101, 
    "Le Kef": 115, "Kalâat Snan": 95, "Kef Est": 100, "Djerissa": 41,
    "Tajerouine": 198, "Es-Sers": 57, "Mahdia": 117, "Chebba": 29,
    "Ksour Essef": 106, "El Jem": 47, "Djerba - Midoun": 39,
    "Djerba-Houmt Souk": 40, "Médenine": 140, "Ben Gardane": 8,
    "Médenine Sud": 142, "Zarzis": 210, "Médenine Nord": 141,
    "Monastir": 135, "Sayada-Lamta-Bou Hajar": 168, "Ksar Hellal": 104,
    "Moknine": 134, "Jemmal": 88, "Sahline": 163, "Ouerdanine": 151,
    "Bekalta": 6, "Téboulba": 205, "Beni Hassen": 10, "Bembla": 7,
    "Ksibet el-Médiouni": 105, "Sidi Bouzid Ouest": 179, 
    "Sidi Bouzid Est": 178, "Regueb": 156, "Sidi Bouzid": 185,
    "Sidi Ali Ben Aoun": 175, "Souk Jedid": 190, "Bir El Hafey": 11,
    "Siliana": 186, "Siliana Nord": 187, "Siliana Sud": 188,
    "Makthar": 119, "Sahloul": 164, "Hammam Sousse": 81, 
    "Zaouit-Ksibat Thrayett": 209, "Hergla": 85, "Sousse": 191,
    "Akouda": 2, "Sousse Jawhara": 192, "Kantaoui": 96, 
    "Sousse Riadh": 194, "Sousse Sidi Abdelhamid": 195, 
    "Sousse Médina": 193, "M'saken": 116, "Bouficha": 19, 
    "Kalaâ Kebira": 93, "Enfidha": 55, "Sidi Thabet": 184,
    "Unknown": 207, 
}

# Define the app
st.title("🏡 Tunisian Property Price Predictor")

# Tabs structure
tab1, tab2, tab3 = st.tabs(["🏠 Input Features", "📊 Prediction Results", "📈 Visualizations"])

with tab1:
    st.header("Enter Property Features:")

    room_count = st.slider("Number of Rooms 🛏️", 1, 18, 3, help="Select the number of rooms in the property.")
    bathroom_count = st.slider("Number of Bathrooms 🛁", 0, 10, 1, help="Select the number of bathrooms.")

    size = st.number_input(
        "Property Size (in m²)", 
        min_value=20.0, 
        max_value=2000.0, 
        value=200.0, 
        step=10.0
    )

    selected_category = st.selectbox("Category", list(category_mapping.keys()))
    selected_region = st.selectbox("Region", list(region_mapping.keys()))

    category_numeric = category_mapping[selected_category]
    region_numeric = region_mapping[selected_region]

    size_per_room = size / room_count if room_count > 0 else 0
    size_per_bathroom = size / bathroom_count if bathroom_count > 0 else 0
    region_cluster = 0  # Replace with actual clustering if applicable

    input_data = np.array([[room_count, bathroom_count, size, category_numeric, region_numeric, size_per_room, size_per_bathroom, region_cluster]])

    if st.button("Predict Price Category"):
        with st.spinner('Processing...'):
            time.sleep(3)  # Simulating a delay
        # Model Prediction
        prediction = xgb_loaded_model.predict(input_data)
        decoded_prediction = ['Low Price', 'Mid Price', 'High Price'][int(prediction[0])]
        st.success(f"Predicted Price Category: {decoded_prediction}")

with tab2:
    # Display Prediction Confidence
    st.header("Prediction Confidence")
    prediction_probabilities = xgb_loaded_model.predict_proba(input_data)[0]
    categories = ['Low Price', 'Mid Price', 'High Price']
    st.bar_chart(pd.DataFrame({'Category': categories, 'Confidence': prediction_probabilities}).set_index('Category'))

with tab3:
    # Feature Contribution Visualization
    st.header("Feature Importance in Prediction")
    feature_importances = xgb_loaded_model.feature_importances_
    feature_names = ['room_count', 'bathroom_count', 'size', 'category_numeric', 
                     'region_numeric', 'size_per_room', 'size_per_bathroom', 'region_cluster']
    st.bar_chart(pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).set_index('Feature'))

     
