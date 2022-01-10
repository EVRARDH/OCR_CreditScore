# Packages
import io
import os
import math
import pickle
import warnings
import pandas as pd
import streamlit as st
import altair as alt

from urllib.request import urlopen
from fastapi.testclient import TestClient
from main import app
from _dashboard_tools import prepare_data, explain_pred

# Setup
PATH_DATA = r'C:\Users\Hugo\Documents\GitHub\OCR_CreditScore\Notebooks'
warnings.filterwarnings("ignore")
client_api = TestClient(app)

# Définition de la fonction contactant l'API
@st.cache
def call_client_api(dict_value):
    response = client_api.post(
        "/prediction",
        json = dict_value
        )
    return response.json()

# Load test dataset
test_set = pd.read_csv(
    os.path.join(PATH_DATA, "test_set_with_preds.csv"),
    )
# Transformation du jeu test
test_set = test_set.drop(columns=["TARGET"])
# Set SK_ID_CURR as index for df and test_set
test_set.set_index('SK_ID_CURR' ,inplace=True)

# 3. Dashboard
# 3.1. Layout
st.set_page_config(layout="wide")

# 3.2. Titre et présentation
from PIL import Image
my_page = urlopen("https://consent.trustarc.com/get?name=oc_logo.png")
# create an image file object
my_picture = io.BytesIO(my_page.read())
img = Image.open(my_picture)
st.image(img, width=300)
# Titre et texte de présentation du dashboard
col_title, col_text = st.columns((2))

with col_title:
    st.title("CSD : Credit Score Dashboard")

with col_text:
    st.write(
    """
    ##
    Ce dashboard va permettre à votre équipe de contacter le modèle de prédiction,
    à travers l'API, afin de savoir si le client est solvable ou non.
    Aussi, ce dashboard présente des graphiques permettant de comprendre et 
    d'expliquer la prédiction du modèle.
    """)

# 3.3 Panneau des variables
# Définition des variables du client
col_variables, col_empty, col_graphs = st.columns([1, 0.5, 4])
with col_variables:
    # Template du bouton "prediction"
    pred_button = st.button("Faire la prédiction")
    # Selectbox avec les identifiants clients
    client_id = st.selectbox(label = "Identifiant du client", options = test_set.index)
    if client_id:
        client_values = {
            "AMT_INCOME_TOTAL": st.number_input(
                label = "Revenu total",
                value = test_set.loc[client_id]["AMT_INCOME_TOTAL"]  
            ), 
            "AMT_CREDIT": st.number_input(
                label = "Valeur des crédits",
                value = test_set.loc[client_id]["AMT_CREDIT"]
            ),
            "AMT_ANNUITY": st.number_input(
                label = "Rente",
                value = test_set.loc[client_id]["AMT_ANNUITY"]
            ),
            "AMT_GOODS_PRICE": st.number_input(
                label = "Prix des biens",
                value = test_set.loc[client_id]["AMT_GOODS_PRICE"]
            ),
            "REGION_POPULATION_RELATIVE": st.number_input(
                label = "Population dans la région",
                value = test_set.loc[client_id]["REGION_POPULATION_RELATIVE"]
            ),
            "DAYS_BIRTH": st.number_input(
                label = "Jour de naissance",
                value = test_set.loc[client_id]["DAYS_BIRTH"]
            ),
            "DAYS_EMPLOYED": st.number_input(
                label = "Jour employé",
                value = test_set.loc[client_id]["DAYS_EMPLOYED"]
            ),
            "DAYS_REGISTRATION": st.number_input(
                label = "Jour d'inscription",
                value = test_set.loc[client_id]["DAYS_REGISTRATION"]
            ),
            "DAYS_ID_PUBLISH": st.number_input(
                label = "Jour de publication de l'ID",
                value = test_set.loc[client_id]["DAYS_ID_PUBLISH"]
            ),
            "CNT_FAM_MEMBERS": st.number_input(
                label = "Membre FAM",
                value = test_set.loc[client_id]["CNT_FAM_MEMBERS"]
            ),
            "HOUR_APPR_PROCESS_START": st.number_input(
                label = "Heure de début du processus",
                value = test_set.loc[client_id]["HOUR_APPR_PROCESS_START"]
            ),
            "EXT_SOURCE_1": st.number_input(
                label = "Données normalisées de sources externes (1)",
                value = test_set.loc[client_id]["EXT_SOURCE_1"]
            ),
            "EXT_SOURCE_2": st.number_input(
                label = "Données normalisées de sources externes (2)",
                value = test_set.loc[client_id]["EXT_SOURCE_2"]
            ),
            "EXT_SOURCE_3": st.number_input(
                label = "Données normalisées de sources externes (3)",
                value = test_set.loc[client_id]["EXT_SOURCE_3"]
            ),
            "OBS_30_CNT_SOCIAL_CIRCLE": st.number_input(
                label = "Cercle social (30)",
                value = test_set.loc[client_id]["OBS_30_CNT_SOCIAL_CIRCLE"]
            ),
            "OBS_60_CNT_SOCIAL_CIRCLE": st.number_input(
                label = "Cercle social (60)",
                value = test_set.loc[client_id]["OBS_60_CNT_SOCIAL_CIRCLE"]
            ),
            "DAYS_LAST_PHONE_CHANGE": st.number_input(
                label = "Jour où le client à changé son portable",
                value = test_set.loc[client_id]["DAYS_LAST_PHONE_CHANGE"]
            ),
            "AMT_REQ_CREDIT_BUREAU_YEAR": st.number_input(
                label = "Crédit bureau par an",
                value = test_set.loc[client_id]["AMT_REQ_CREDIT_BUREAU_YEAR"]
            ),
            "PROPORTION_LIFE_EMPLOYED": st.number_input(
                label = "Proportion de la vie active",
                value = test_set.loc[client_id]["PROPORTION_LIFE_EMPLOYED"]
            ),
            "INCOME_TO_CREDIT_RATIO": st.number_input(
                label = "Ratio revenu/crédit",
                value = test_set.loc[client_id]["INCOME_TO_CREDIT_RATIO"]
            ),
            "INCOME_TO_ANNUITY_RATIO": st.number_input(
                label = "Ratio revenu/rente",
                value = test_set.loc[client_id]["INCOME_TO_ANNUITY_RATIO"]
            ),
            "INCOME_TO_ANNUITY_RATIO_BY_AGE": st.number_input(
                label = "Ratio revenu/rente en fonction de l'âge",
                value = test_set.loc[client_id]["INCOME_TO_ANNUITY_RATIO_BY_AGE"]
            ),
            "CREDIT_TO_ANNUITY_RATIO": st.number_input(
                label = "Ratio crédit/rente",
                value = test_set.loc[client_id]["CREDIT_TO_ANNUITY_RATIO"]
            ),
            "CREDIT_TO_ANNUITY_RATIO_BY_AGE": st.number_input(
                label = "Ratio crédit/rente en fonction de l'âge",
                value = test_set.loc[client_id]["CREDIT_TO_ANNUITY_RATIO_BY_AGE"]
            ),
            "INCOME_TO_FAMILYSIZE_RATIO": st.number_input(
                label = "Ratio revenu/taille de la famile",
                value = test_set.loc[client_id]["INCOME_TO_FAMILYSIZE_RATIO"]
            ),
            "WEEKDAY_APPR_PROCESS_START_1": st.number_input(
                label = "Jour du début de processus (1)",
                value = test_set.loc[client_id]["WEEKDAY_APPR_PROCESS_START_1"]
            ),
            "WEEKDAY_APPR_PROCESS_START_2": st.number_input(
                "Jour du début de processus (2)",
                value = test_set.loc[client_id]["WEEKDAY_APPR_PROCESS_START_2"]
            )
        }

# En cas de problème avec les valeurs manquantes dans le dictionnaire 'client_values' 
for key in client_values.keys():
    if math.isnan(client_values[key]):
        client_values[key] = 0

# 3.4. Panneau avec résultats et graphiques
if pred_button:
    with col_graphs:
        container1 = st.container()
        container2 = st.container()
        # Premier résultat (Prédiction depuis l'API)
        with container1:
            st.markdown(
                "<h2 style='text-align: left; color: black; size:2;'>Prédiction depuis l'API</h2>",
                unsafe_allow_html=True
                )
            # Résultats sous forme de phrase
            results = "Le modèle nous informe que le client est %s avec une probabilité de %s" % (
                    call_client_api(client_values)["Prediction"].lower(),
                    str(call_client_api(client_values)["Probabilité"])
            )
            st.write(results)
            # Graphique
            results = [
            [call_client_api(client_values)["Prediction"],
            call_client_api(client_values)["Probabilité"]]
            ]
            data = pd.DataFrame(results,columns=["Prédiction", "Probabilité"])
            if data["Prédiction"][0] == "Solvable":
                results = [
                ["Non solvable",
                1-float(data["Probabilité"][0])]
                ]
                new_data = pd.DataFrame(
                    results,
                    columns=["Prédiction", "Probabilité"]
                )
                data = data.append(new_data)
            elif data["Prédiction"][0] == "Non solvable":
                results = [
                ["Solvable",
                1-float(data["Probabilité"][0])]
                ]
                new_data = pd.DataFrame(
                    results,
                    columns=["Prédiction", "Probabilité"]
                )
                data = data.append(new_data)
            graph = alt.Chart(data).mark_bar().encode(
                x = "Probabilité",
                y = "Prédiction",
                color = "Prédiction"
            )
            st.altair_chart(graph, use_container_width=True)
        # Second sous-titre
        with container2:
            # Titre
            st.markdown(
                "<h2 style='text-align: left; color: black; size:2;'>Explication de la prédiction</h2>",
                unsafe_allow_html=True
                )
            st.write("Voici la liste des variables qui expliquent le plus la prédiction du modèle")
            # Graph (Importance des features pour un client spécifique)
            new_test_set = prepare_data(client_values)
            st.pyplot(explain_pred(new_test_set))
