import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
import base64

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =========================
# SIDEBAR COLOR STYLING
# =========================
st.markdown("""
<style>

[data-testid="stSidebar"]{
    background: linear-gradient(180deg, #0F172A, #1E3A8A);
}

[data-testid="stSidebar"] *{
    color: white;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{
    color: #FFD700;
    font-weight: bold;
}

.stRadio label{
    color: white !important;
    font-size: 16px !important;
}

.stButton > button{
    background-color: #10B981;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}

.stButton > button:hover{
    background-color: #059669;
    color: white;
}

.stSelectbox div[data-baseweb="select"]{
    background-color: white;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stSelectbox div[data-baseweb="select"] > div {
    color: black !important;
    background-color: white !important;
}
div[role="listbox"] ul {
    background-color: white !important;
}
div[role="option"] {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)


# ---------- Background Image ----------
def add_bg_from_local(image_file):
    try:
        image_path = os.path.join(os.path.dirname(__file__), image_file)
        if not os.path.exists(image_path):
            return
        with open(image_path, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        pass

add_bg_from_local("disease_bg.jpg")


# ===============================
# SESSION STATE INIT
# ===============================
defaults = {
    "logged_in": False,
    "username": "",
    "df": None,
    "cleaned_df": None,
    "encoded_df": None,         # NEW: stores the label-encoded feature dataframe
    "model": None,
    "features": None,
    "encoders": {},             # column -> LabelEncoder for ALL feature columns (str + numeric)
    "target_encoder": None,
    "target_name": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ===============================
# DATABASE
# ===============================
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")
conn.commit()


# ===============================
# TITLE
# ===============================
st.title("Disease Prediction and Diagnosis Platform")


# ===============================
# LOGIN / REGISTER
# ===============================
if not st.session_state.logged_in:

    menu = st.sidebar.radio("Menu", ["Register", "Login"])

    if menu == "Register":
        st.subheader("Create Account")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Register"):
            if not user or not pwd:
                st.error("Username and password cannot be empty")
            else:
                try:
                    cursor.execute(
                        "INSERT INTO users(username,password) VALUES (?,?)",
                        (user, pwd)
                    )
                    conn.commit()
                    st.success("Registered Successfully")
                except sqlite3.IntegrityError:
                    st.error("Username already exists")

    if menu == "Login":
        st.subheader("Login")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            cursor.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (user, pwd)
            )
            result = cursor.fetchone()
            if result:
                st.session_state.logged_in = True
                st.session_state.username = user
                st.rerun()
            else:
                st.error("Invalid credentials")


# ======================
# DISEASE ALERT BUTTON
# ======================
st.sidebar.markdown("---")

if st.sidebar.button("🚨 Disease Alert Guide"):
    st.sidebar.info("""
🔹 FLU
• Symptoms: Fever, Cough, Fatigue
• Recommended: Rest, Hydration, Medication

🔹 COMMON COLD
• Symptoms: Runny Nose, Sore Throat, Headache
• Recommended: Rest and Fluids

🔹 BRONCHITIS
• Symptoms: Cough, Shortness of Breath, Fatigue
• Recommended: Medical Consultation

🔹 PNEUMONIA
• Symptoms: High Fever, Shortness of Breath, Fatigue
• Recommended: Immediate Medical Attention

🔹 HEALTHY STATUS
• Maintain balanced diet
• Exercise regularly
• Sleep 7–8 hours daily
""")


# ===============================
# HELPER: DATA CLEANING
# ===============================
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Try to coerce object columns that look numeric
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() >= 0.8 * len(df[col].dropna()) and converted.notna().any():
                df[col] = converted

    # Fill missing values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
        else:
            if df[col].isnull().any():
                mode_vals = df[col].mode(dropna=True)
                fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
                df[col] = df[col].fillna(fill_val)
            df[col] = df[col].astype(str)

    return df


# ===============================
# KEY CHANGE: LABEL ENCODE ALL FEATURE COLUMNS
# (Both categorical AND numeric columns get a LabelEncoder)
# ===============================
def label_encode_all_features(X: pd.DataFrame):
    """
    Apply LabelEncoder to EVERY feature column regardless of dtype.
    - String/categorical columns: encode category names → integers.
    - Numeric columns: encode unique numeric values → integers
      (preserves ordinal mapping; all columns end up as int).
    Returns:
        X_encoded : pd.DataFrame  (all columns are int64)
        encoders  : dict { col_name -> fitted LabelEncoder }
    """
    X = X.copy()

    # Remove duplicate column names if any
    if X.columns.duplicated().any():
        X = X.loc[:, ~X.columns.duplicated()]

    encoders = {}
    for col in X.columns:
        le = LabelEncoder()
        # Convert to string first so LabelEncoder handles all dtypes uniformly
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    return X.astype(int), encoders


# ===============================
# AFTER LOGIN
# ===============================
if st.session_state.logged_in:

    st.sidebar.success(f"Logged in as {st.session_state.username}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    menu = st.sidebar.radio(
        "Select Option",
        ["Dataset Upload", "Train Model", "Prediction"]
    )

    # ===========================
    # DATA UPLOAD
    # ===========================
    if menu == "Dataset Upload":

        file = st.file_uploader("Upload CSV", type=["csv"])

        if file:
            df = pd.read_csv(file)
            st.session_state.df = df

            st.subheader("Full Dataset")
            st.dataframe(df)

            st.subheader("Head (10 rows)")
            st.dataframe(df.head(10))

            st.subheader("Missing Values (before cleaning)")
            st.write(df.isnull().sum())

            cleaned = clean_dataframe(df)
            st.success("Missing values handled")

            dup = cleaned.duplicated().sum()
            st.write("Duplicates found:", dup)
            cleaned.drop_duplicates(inplace=True)
            st.success("Duplicates removed")

            st.subheader("Missing Values (after cleaning)")
            st.write(cleaned.isnull().sum())

            st.session_state.cleaned_df = cleaned

    # ===========================
    # TRAIN MODEL
    # ===========================
    if menu == "Train Model":

        if st.session_state.cleaned_df is None:
            st.warning("Upload dataset first")

        else:
            df = st.session_state.cleaned_df.copy()

            target = st.selectbox("Select Target Column", df.columns)

            if st.button("Train Model"):

                df = df.dropna(subset=[target])

                X = df.drop(columns=[target])
                y = df[target]

                # --------------------------------------------------
                # LABEL ENCODE ALL FEATURE COLUMNS (the key change)
                # --------------------------------------------------
                X_encoded, encoders = label_encode_all_features(X)

                # Show encoding summary in an expander
                with st.expander("📊 Label Encoding Summary (all feature columns)"):
                    for col, le in encoders.items():
                        mapping = {orig: enc for enc, orig in enumerate(le.classes_)}
                        st.write(f"**{col}**: {mapping}")

                # Store encoded df for reference
                st.session_state.encoded_df = X_encoded

                # ---- Encode TARGET if it's categorical ----
                target_encoder = None
                if y.dtype == "object" or not pd.api.types.is_numeric_dtype(y):
                    target_encoder = LabelEncoder()
                    y = target_encoder.fit_transform(y.astype(str))
                    if target_encoder is not None:
                        st.info(f"Target '{target}' encoding: "
                                f"{ {orig: enc for enc, orig in enumerate(target_encoder.classes_)} }")

                if X_encoded.empty or len(X_encoded) < 5:
                    st.error("Not enough clean data to train. Please check your dataset.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_encoded, y, test_size=0.2, random_state=42
                    )

                    model = RandomForestClassifier(random_state=42)
                    model.fit(X_train, y_train)

                    pred = model.predict(X_test)

                    acc  = accuracy_score(y_test, pred)
                    prec = precision_score(y_test, pred, average="weighted", zero_division=0)
                    rec  = recall_score(y_test, pred, average="weighted", zero_division=0)
                    f1   = f1_score(y_test, pred, average="weighted", zero_division=0)

                    st.success("✅ Model Trained Successfully!")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy",  f"{acc:.4f}")
                    col2.metric("Precision", f"{prec:.4f}")
                    col3.metric("Recall",    f"{rec:.4f}")
                    col4.metric("F1 Score",  f"{f1:.4f}")

                    joblib.dump(model, "disease_model.pkl")

                    st.session_state.model          = model
                    st.session_state.features       = list(X_encoded.columns)
                    st.session_state.encoders       = encoders
                    st.session_state.target_encoder = target_encoder
                    st.session_state.target_name    = target

    # ===========================
    # PREDICTION
    # ===========================
    if menu == "Prediction":

        if st.session_state.model is None:
            st.warning("Train model first")

        else:
            st.subheader("Enter Values for Prediction")

            inputs = {}
            encoders = st.session_state.encoders or {}

            for col in st.session_state.features:
                if col in encoders:
                    le = encoders[col]
                    # Show original labels in the dropdown
                    options = list(le.classes_)
                    choice = st.selectbox(f"{col}", options)
                    # Transform the chosen label back to its encoded integer
                    inputs[col] = int(le.transform([choice])[0])
                else:
                    inputs[col] = st.number_input(col, value=0.0)

            if st.button("Predict"):
                input_df = pd.DataFrame([inputs])[st.session_state.features]
                input_df = input_df.astype(float)

                model      = st.session_state.model
                prediction = model.predict(input_df)[0]

                # Decode target back to original label
                if st.session_state.target_encoder is not None:
                    prediction_label = st.session_state.target_encoder.inverse_transform(
                        [int(prediction)]
                    )[0]
                else:
                    prediction_label = prediction

                st.success(f"🩺 Predicted Disease: **{prediction_label}**")

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_df)[0]
                    max_prob = max(probabilities)
                    st.write(f"🎯 Prediction Confidence: **{max_prob * 100:.2f}%**")

                    # Show all class probabilities
                    with st.expander("📈 All Class Probabilities"):
                        classes = model.classes_
                        target_enc = st.session_state.target_encoder
                        prob_data = {}
                        for cls, prob in zip(classes, probabilities):
                            if target_enc is not None:
                                label = target_enc.inverse_transform([int(cls)])[0]
                            else:
                                label = cls
                            prob_data[str(label)] = round(float(prob) * 100, 2)
                        prob_df = pd.DataFrame(
                            list(prob_data.items()),
                            columns=["Disease", "Probability (%)"]
                        ).sort_values("Probability (%)", ascending=False)
                        st.dataframe(prob_df, use_container_width=True)
