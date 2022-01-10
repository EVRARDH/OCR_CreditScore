"""Script pour l'API
"""

# !pip install fastapi

# 1. Import des librairies Python et setup des classes
import os
import pickle
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI

PATH = r'C:\Users\Hugo\Documents\GitHub\OCR_CreditScore\Notebooks'

# 2. Définition de la classe BaseModel
class ClientData(BaseModel):
    AMT_INCOME_TOTAL: float 
    AMT_CREDIT: float 
    AMT_ANNUITY: float
    AMT_GOODS_PRICE : float
    REGION_POPULATION_RELATIVE : float
    DAYS_BIRTH : float
    DAYS_EMPLOYED : float
    DAYS_REGISTRATION : float
    DAYS_ID_PUBLISH : float
    CNT_FAM_MEMBERS : float
    HOUR_APPR_PROCESS_START : float
    EXT_SOURCE_1 : float
    EXT_SOURCE_2 : float
    EXT_SOURCE_3 : float
    OBS_30_CNT_SOCIAL_CIRCLE : float
    OBS_60_CNT_SOCIAL_CIRCLE : float
    DAYS_LAST_PHONE_CHANGE : float
    AMT_REQ_CREDIT_BUREAU_YEAR : float
    PROPORTION_LIFE_EMPLOYED : float
    INCOME_TO_CREDIT_RATIO : float
    INCOME_TO_ANNUITY_RATIO : float
    INCOME_TO_ANNUITY_RATIO_BY_AGE : float
    CREDIT_TO_ANNUITY_RATIO : float
    CREDIT_TO_ANNUITY_RATIO_BY_AGE : float
    INCOME_TO_FAMILYSIZE_RATIO : float
    WEEKDAY_APPR_PROCESS_START_1 : float
    WEEKDAY_APPR_PROCESS_START_2 : float

app = FastAPI()

# 3. Prédiction
@app.post('/prediction')
async def predict_score(client: ClientData):
    # Get data
    data = client.dict()
    # Load model
    model =  pickle.load(open(os.path.join(PATH, 'banking_model_bank_api.md'), 'rb'))
    data_in = [[
            data['AMT_INCOME_TOTAL'], data['AMT_CREDIT'], data['AMT_ANNUITY'], data['AMT_GOODS_PRICE'], 
            data['REGION_POPULATION_RELATIVE'], data['DAYS_BIRTH'], data['DAYS_EMPLOYED'], data['DAYS_REGISTRATION'], 
            data['DAYS_ID_PUBLISH'], data['CNT_FAM_MEMBERS'], data['HOUR_APPR_PROCESS_START'], data['EXT_SOURCE_1'], 
            data['EXT_SOURCE_2'], data['EXT_SOURCE_3'], data['OBS_30_CNT_SOCIAL_CIRCLE'], data['OBS_60_CNT_SOCIAL_CIRCLE'], 
            data['DAYS_LAST_PHONE_CHANGE'], data['AMT_REQ_CREDIT_BUREAU_YEAR'], data['PROPORTION_LIFE_EMPLOYED'], 
            data['INCOME_TO_CREDIT_RATIO'], data['INCOME_TO_ANNUITY_RATIO'], data['INCOME_TO_ANNUITY_RATIO_BY_AGE'], 
            data['CREDIT_TO_ANNUITY_RATIO'], data['CREDIT_TO_ANNUITY_RATIO_BY_AGE'], data['INCOME_TO_FAMILYSIZE_RATIO'], 
            data['WEEKDAY_APPR_PROCESS_START_1'], data['WEEKDAY_APPR_PROCESS_START_2']
        ]]
    prediction = model.predict(data_in)
    probability = model.predict_proba(data_in).max()
    if prediction == 0 and probability > 0.90:
        prediction = "Solvable"
    elif prediction == 0 and probability < 0.90:
        prediction = "Non solvable"
    else:
        prediction = "Non solvable"

    return {
        "Prediction": prediction,
        "Probabilité": round(probability, 3)
    }

# 4. Start API avec uvicorn on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(
        app,
        host='127.0.0.1',
        port=8000,
        log_level='info',
        reload=True
        )
