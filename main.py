import pandas as pd
import numpy as np
import joblib
import chromadb
import re
import warnings

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str


# =========================
# LOAD MODELS
# =========================

print("Loading models...")

pred_dict = joblib.load("models/ocean_prediction_pipeline.pkl")
prediction_model = pred_dict["prediction_model"]
prediction_scaler = pred_dict["scaler"]

cluster_dict = joblib.load("models/ocean_clustering_pipeline.pkl")
cluster_model = cluster_dict["cluster_model"]
cluster_scaler = cluster_dict["scaler"]

print("Models loaded")


# =========================
# FEATURE SIZE
# =========================

PRED_FEATURES = prediction_scaler.n_features_in_
CLUSTER_FEATURES = cluster_scaler.n_features_in_

print("Prediction model expects:", PRED_FEATURES)
print("Cluster model expects:", CLUSTER_FEATURES)


# =========================
# CONNECT CHROMADB
# =========================

try:

    client = chromadb.PersistentClient(path="./argo_chroma_db")
    collection = client.get_collection(name="argo_ocean_data")

    print("ChromaDB connected")
    print("Total records:", collection.count())

except:

    collection = None
    print("ChromaDB failed")


# =========================
# EXTRACT NUMBERS
# =========================

def extract_numbers(text):

    numbers = re.findall(r"-?\d+\.\d+|-?\d+", text)

    return [float(x) for x in numbers]


# =========================
# CHROMADB SEARCH
# =========================

def chroma_search(query):

    if collection is None:
        return None

    try:

        results = collection.query(
            query_texts=[query],
            n_results=50
        )

        docs = results["documents"][0]

        rows = []

        for doc in docs:

            nums = extract_numbers(doc)

            if len(nums) >= 5:
                rows.append(nums[:5])

        if len(rows) == 0:
            return None

        df = pd.DataFrame(rows)

        df.columns = [
            "pressure",
            "temperature",
            "salinity",
            "latitude",
            "longitude"
        ]

        return df

    except Exception as e:

        print("Chroma search error:", e)

        return None


# =========================
# MODEL RUNNER
# =========================

def run_models(data):

    try:

        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.dropna()

        results = []

        if data.shape[1] >= PRED_FEATURES:

            Xp = data.iloc[:, :PRED_FEATURES].values
            Xp = prediction_scaler.transform(Xp)

            preds = prediction_model.predict(Xp)

            results.append(f"Prediction score: {np.mean(preds):.2f}")

        if data.shape[1] >= CLUSTER_FEATURES:

            Xc = data.iloc[:, :CLUSTER_FEATURES].values
            Xc = cluster_scaler.transform(Xc)

            try:
                clusters = cluster_model.fit_predict(Xc)
            except:
                clusters = cluster_model.predict(Xc)

            clusters = list(map(int, np.unique(clusters)))
            results.append(f"Clusters detected: {clusters}")

        return results

    except Exception as e:

        return [f"Model processing error: {str(e)}"]


# =========================
# ANOMALY DETECTOR
# =========================

def detect_anomalies(df):

    temp_anomaly = df[
        abs(df["temperature"] - df["temperature"].mean())
        > df["temperature"].std()
    ]

    sal_anomaly = df[
        abs(df["salinity"] - df["salinity"].mean())
        > df["salinity"].std()
    ]

    pressure_anomaly = df[
        abs(df["pressure"] - df["pressure"].mean())
        > df["pressure"].std()
    ]

    return len(temp_anomaly), len(sal_anomaly), len(pressure_anomaly)

# =========================
# HEATWAVE DETECTOR
# =========================

def detect_heatwave(df):

    threshold = df["temperature"].mean() + df["temperature"].std()

    heatwave_df = df.copy()

    heatwave_df["heatwave_signal"] = np.where(
        heatwave_df["temperature"] > threshold,
        "YES",
        "NO"
    )

    return heatwave_df.head(10)
# =========================
# CYCLONE DETECTOR
# =========================

def detect_cyclone(df):

    pressure_threshold = df["pressure"].mean() - df["pressure"].std()
    temp_threshold = df["temperature"].mean()

    cyclone_df = df.copy()

    cyclone_df["cyclone_signal"] = np.where(
        (cyclone_df["pressure"] < pressure_threshold) &
        (cyclone_df["temperature"] > temp_threshold),
        "YES",
        "NO"
    )

    return cyclone_df.head(10)
# =========================
# OCEAN INSTABILITY
# =========================

def detect_instability(df):

    instability_df = df.copy()

    temp_var = df["temperature"].var()
    sal_var = df["salinity"].var()

    threshold = temp_var + sal_var

    instability_df["instability_index"] = (
        abs(instability_df["temperature"] - df["temperature"].mean()) +
        abs(instability_df["salinity"] - df["salinity"].mean())
    )

    instability_df["instability_signal"] = np.where(
        instability_df["instability_index"] > threshold,
        "HIGH",
        "NORMAL"
    )

    return instability_df.head(10)
# =========================
# NEW AI FEATURES
# =========================

def greet_user():

    return """
🌊 Hello! I am FloatChat Ocean AI.

I analyze ARGO float ocean data using machine learning.

You can ask things like:

• ocean temperature
• salinity anomaly
• ocean clusters
• pressure analysis
• latitude / longitude range
• ocean trends
• explain ocean anomalies
"""


def explain_ocean_science(question):

    q = question.lower()

    if "latitude" in q and "longitude" in q:

        return """
🌍 Latitude and Longitude in Oceanography

Latitude measures how far north or south a location is from the equator.

Longitude measures how far east or west a location is from the Prime Meridian.

Ocean scientists use these coordinates to track ARGO floats and study ocean currents.
"""

    if "why salinity changes" in q:

        return """
🧂 Why Ocean Salinity Changes

Salinity changes due to:

• evaporation  
• rainfall  
• river inflow  
• melting ice  
• ocean circulation
"""

    if "why temperature changes" in q:

        return """
🌡 Why Ocean Temperature Changes

Ocean temperature varies because of:

• solar radiation  
• ocean currents  
• seasonal weather  
• depth of water  
• climate patterns
"""

    if "what is anomaly" in q:

        return """
🚨 Ocean Anomaly

An anomaly is a value that deviates significantly from the normal range.

Example:
If average temperature is 27°C but one reading is 32°C, that could be an anomaly.
"""

    return None


# =========================
# DATASET SUMMARY
# =========================

def dataset_summary(df):

    return f"""
📊 Ocean Dataset Summary

Records analyzed: {len(df)}

Temperature range:
{df['temperature'].min():.2f} °C to {df['temperature'].max():.2f} °C

Salinity range:
{df['salinity'].min():.2f} to {df['salinity'].max():.2f}

Latitude range:
{df['latitude'].min():.2f} to {df['latitude'].max():.2f}

Longitude range:
{df['longitude'].min():.2f} to {df['longitude'].max():.2f}
"""
# =========================
# NATURAL LANGUAGE ROUTER

# =========================
# =========================
# QUERY NORMALIZER
# =========================

def normalize_query(query):

    q = query.lower()

    replacements = {

        "sea temperature": "temperature",
        "water temperature": "temperature",
        "ocean warming": "temperature",
        "sea warming": "temperature",

        "salt level": "salinity",
        "saltiness": "salinity",

        "water pressure": "pressure",
        "deep pressure": "pressure",

        "ocean patterns": "cluster",
        "cluster patterns": "cluster",

        "argo float": "float",
        "float data": "summary",
        "ocean data": "summary",

        "abnormal temperature": "temperature anomaly",
        "abnormal salinity": "salinity anomaly"
    }

    for key, value in replacements.items():

        if key in q:
            return value

    return q
def smart_query_router(query):

    q = query.lower()

    if "heatwave" in q:
        return "heatwave"

    if "cyclone" in q or "storm" in q:
        return "cyclone"

    if "instability" in q or "unstable" in q:
        return "instability"

    if any(word in q for word in [
        "anomaly","abnormal","unusual","outlier"
    ]):
        return "anomaly"

    if any(word in q for word in [
        "temperature","temp","warming","heat"
    ]):
        return "temperature"

    if any(word in q for word in [
        "salinity","salt"
    ]):
        return "salinity"

    if "pressure" in q:
        return "pressure"

    if "cluster" in q or "pattern" in q:
        return "cluster"

    if "latitude" in q:
        return "latitude"

    if "longitude" in q:
        return "longitude"

    return "general"


# =========================
# QUERY ANALYZER
# =========================

def analyze_query(query):

    query = normalize_query(query)
    q = query.lower()

    # greetings
    if q in ["hi", "hello", "hey"]:
        return greet_user()

    # scientific explanation
    explanation = explain_ocean_science(q)
    if explanation:
        return explanation

    # NEW SMART ROUTER
    intent = smart_query_router(query)

    rag_data = chroma_search(query)

    if rag_data is None:
        return "No ocean records were found."

    df = rag_data.copy()

    # dataset summary
    if "summary" in q or "dataset" in q:
        return dataset_summary(df)


    # =========================
    # HEATWAVE DETECTION
    # =========================
    if intent == "heatwave":

        threshold = df["temperature"].mean() + df["temperature"].std()

        heatwave_df = df.copy()

        heatwave_df["heatwave_signal"] = np.where(
            heatwave_df["temperature"] > threshold,
            "YES",
            "NO"
        )

        heatwave_count = (heatwave_df["heatwave_signal"] == "YES").sum()
        total = len(heatwave_df)

        sample = heatwave_df[["temperature","latitude","longitude","heatwave_signal"]].head(5)

        return f"""
🌡 Marine Heatwave Analysis

Heatwave threshold: {threshold:.2f} °C

Heatwave signals detected: {heatwave_count} / {total}

Interpretation:
{"⚠️ Possible marine heatwave detected." if heatwave_count > 0 else "✅ No marine heatwave detected in analyzed records."}

Sample observations:

{sample.to_string(index=False)}
"""


    # =========================
    # CYCLONE DETECTION
    # =========================
    if intent == "cyclone":

        pressure_threshold = df["pressure"].mean() - df["pressure"].std()
        temp_threshold = df["temperature"].mean()

        cyclone_df = df.copy()

        cyclone_df["cyclone_signal"] = np.where(
            (cyclone_df["pressure"] < pressure_threshold) &
            (cyclone_df["temperature"] > temp_threshold),
            "YES",
            "NO"
        )

        cyclone_count = (cyclone_df["cyclone_signal"] == "YES").sum()
        total = len(cyclone_df)

        sample = cyclone_df[["pressure","temperature","latitude","cyclone_signal"]].head(5)

        return f"""
🌪 Cyclone Signal Analysis

Cyclone conditions checked:
• Low ocean pressure
• Warm sea surface temperature

Cyclone signals detected: {cyclone_count} / {total}

Interpretation:
{"⚠️ Conditions favorable for cyclone formation." if cyclone_count > 0 else "✅ No cyclone formation signals detected."}

Sample observations:

{sample.to_string(index=False)}
"""


    # =========================
    # OCEAN INSTABILITY
    # =========================
    if intent == "instability":

        instability_df = df.copy()

        temp_var = df["temperature"].var()
        sal_var = df["salinity"].var()

        threshold = temp_var + sal_var

        instability_df["instability_index"] = (
            abs(instability_df["temperature"] - df["temperature"].mean()) +
            abs(instability_df["salinity"] - df["salinity"].mean())
        )

        instability_df["instability_signal"] = np.where(
            instability_df["instability_index"] > threshold,
            "HIGH",
            "NORMAL"
        )

        high_instability = (instability_df["instability_signal"] == "HIGH").sum()
        total = len(instability_df)

        sample = instability_df[["temperature","salinity","instability_signal"]].head(5)

        return f"""
🌊 Ocean Instability Analysis

High instability signals: {high_instability} / {total}

Interpretation:
{"⚠️ Ocean instability detected." if high_instability > 0 else "✅ Ocean conditions appear stable."}

Sample observations:

{sample.to_string(index=False)}
"""


    # latitude
    if intent == "latitude":

        return f"""
📍 Ocean Latitude Analysis

Average latitude: {df['latitude'].mean():.2f}

Minimum latitude: {df['latitude'].min():.2f}

Maximum latitude: {df['latitude'].max():.2f}
"""


    # longitude
    if intent == "longitude":

        return f"""
📍 Ocean Longitude Analysis

Average longitude: {df['longitude'].mean():.2f}

Minimum longitude: {df['longitude'].min():.2f}

Maximum longitude: {df['longitude'].max():.2f}
"""


    # temperature
    if intent == "temperature":

        avg = df["temperature"].mean()

        return f"""
🌡 Ocean Temperature

Average temperature: {avg:.2f} °C
Records analyzed: {len(df)}
"""


    # salinity
    if intent == "salinity":

        avg = df["salinity"].mean()

        return f"""
🧂 Ocean Salinity

Average salinity: {avg:.2f}
Records analyzed: {len(df)}
"""


    # pressure
    if intent == "pressure":

        avg = df["pressure"].mean()

        return f"""
🌊 Ocean Pressure

Average pressure: {avg:.2f}
Records analyzed: {len(df)}
"""


    # anomaly
    if intent == "anomaly":

        t, s, p = detect_anomalies(df)

        return f"""
🚨 Ocean Anomaly Report

Temperature anomalies: {t}
Salinity anomalies: {s}
Pressure anomalies: {p}
"""


    # clusters
    if intent == "cluster":

        results = run_models(df)

        return f"""
🧠 Ocean Clustering Analysis

{' '.join(results)}
"""


    # general report
    model_results = run_models(df)

    return f"""
🌊 FLOATCHAT OCEAN AI REPORT

Records analyzed: {len(df)}

Average Temperature: {df['temperature'].mean():.2f}
Average Salinity: {df['salinity'].mean():.2f}
Average Pressure: {df['pressure'].mean():.2f}

{' '.join(model_results)}
"""
# =========================
# API
# =========================

@app.post("/chat")
async def chat(req: ChatRequest):

    response = analyze_query(req.message)

    return {"response": response}


@app.get("/")
async def home():

    return FileResponse("index.html")


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8008)