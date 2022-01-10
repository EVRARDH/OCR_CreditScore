# Packages
import os
import pickle
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

# Répertoire de travail
PATH_DATA = r"C:\Users\Hugo\Documents\GitHub\OCR_CreditScore\Notebooks"
PATH_MODEL = r'C:\Users\Hugo\Documents\GitHub\OCR_CreditScore\API_Dashboard'

# Get data
df = pd.read_csv(
    os.path.join(PATH_DATA, "app_encoded.csv"),
    sep = ";",
    encoding='utf8'
    )
test_set = pd.read_csv(
    os.path.join(PATH_DATA, "test_set_with_preds.csv"),
    )
# Transformation du jeu test
test_set = test_set.drop(columns=["TARGET"])
# Set SK_ID_CURR as index for df and test_set
df.set_index('SK_ID_CURR' ,inplace=True)
test_set.set_index('SK_ID_CURR' ,inplace=True)

# Functions
def prepare_data(dict_values):
        """Préparation des données pour la librairie Lime

        Args:
            dict_values ([type]): Dictionnaire
            provenant du dashboard avec les valeurs
            clients
        """
        # Transform dict_value into Pandas dataframe object
        new_x = pd.DataFrame(dict_values, index=[1])
        # Rewrite test_set with new observation
        new_test_set = test_set.append(new_x, ignore_index=True)

        return new_test_set 

def explain_pred(test_set):
        """Fonction permettant d'éditer le graphique
        des features de Lime
        """
        # Load model
        model_banking_dash =  pickle.load(
            open(
                os.path.join(PATH_MODEL, 'banking_model_bank_dashboard.md'
                ), 'rb')
                )
        # Get NearestNeighbors
        x_test_transformed = pd.DataFrame(
            model_banking_dash[0].transform(test_set),
            columns = test_set.columns,
            index = test_set.index
            )
        # Features importance for one client
        lime1 = LimeTabularExplainer(x_test_transformed,
                                    feature_names=test_set.columns,
                                    class_names=["Solvable", "Non Solvable"],
                                    discretize_continuous=False)                 
        # Compute exp
        exp = lime1.explain_instance(x_test_transformed.iloc[-1],
                                    model_banking_dash.predict_proba,
                                    num_samples=250)

        return exp.as_pyplot_figure()       
