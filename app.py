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

/* Sidebar Background */
[data-testid="stSidebar"]{
    background: linear-gradient(180deg, #0F172A, #1E3A8A);
}

/* Sidebar Text */
[data-testid="stSidebar"] *{
    color: white;
}

/* Sidebar Title */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{
    color: #FFD700;
    font-weight: bold;
}

/* Radio Button Text */
.stRadio label{
    color: white !important;
    font-size: 16px !important;
}

/* Buttons */
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

/* Select Box */
.stSelectbox div[data-baseweb="select"]{
    background-color: white;
    border-radius: 8px;
}

/* Success, Warning, Error */
.stSuccess{
    background-color: rgba(16,185,129,0.2);
}

.stWarning{
    background-color: rgba(245,158,11,0.2);
}

.stError{
    background-color: rgba(239,68,68,0.2);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Selectbox text */
.stSelectbox div[data-baseweb="select"] > div {
    color: black !important;
    background-color: white !important;
}

/* Dropdown options */
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
    """Apply a background image if it exists. Fails silently instead of
    crashing the whole app when the file is missing (e.g. on first deploy)."""
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

            .stContainer {{
                background-color: rgba(255,255,255,0.1);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        # Don't let a missing/corrupt background image take down the app
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
    "model": None,
    "features": None,
    "encoders": {},          # column -> LabelEncoder, for categorical FEATURES
    "target_encoder": None,  # LabelEncoder for the target, if it was categorical
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

    # REGISTER
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

    # LOGIN
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
• Common symptoms:
  Fever, Cough, Fatigue
• Recommended:
  Rest, Hydration, Medication

🔹 COMMON COLD
• Common symptoms:
  Runny Nose, Sore Throat, Headache
• Recommended:
  Rest and Fluids

🔹 BRONCHITIS
• Common symptoms:
  Cough, Shortness of Breath, Fatigue
• Recommended:
  Medical Consultation

🔹 PNEUMONIA
• Common symptoms:
  High Fever, Shortness of Breath, Fatigue
• Recommended:
  Immediate Medical Attention

🔹 HEALTHY STATUS
• Maintain balanced diet
• Exercise regularly
• Sleep 7–8 hours daily
""")


# ===============================
# HELPER: ROBUST DATA CLEANING
# ===============================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a dataframe column-by-column so that no NaNs or mixed types
    remain (this is what was breaking model.fit -> numpy.asarray()).
    - Numeric-looking object columns are converted to numbers.
    - Truly numeric columns: missing values filled with the median.
    - Truly categorical columns: missing values filled with the mode,
      or "Unknown" if no mode exists (e.g. an all-NaN column).
    """
    df = df.copy()

    for col in df.columns:
        # Try to coerce object columns that are "secretly" numeric
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")
            # Only treat it as numeric if most values converted successfully
            if converted.notna().sum() >= 0.8 * len(df[col].dropna()) and converted.notna().any():
                df[col] = converted

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
            # Make sure everything is plain string for consistent encoding later
            df[col] = df[col].astype(str)

    return df


def safe_prepare_features(X: pd.DataFrame):
    """
    Bulletproof feature prep: doesn't trust dtype labels (which can be
    wrong/misleading), instead it actually TRIES to convert each column
    to numbers, and only label-encodes if that genuinely fails.
    Also guards against duplicate column names, which silently break
    column-wise operations like X[col] = ... in pandas.
    """
    X = X.copy()

    # Guard: duplicate column names (e.g. two columns both called "Gender")
    # make X[col] return a 2D DataFrame instead of a Series and break encoding.
    if X.columns.duplicated().any():
        X = X.loc[:, ~X.columns.duplicated()]

    encoders = {}
    for col in X.columns:
        series = X[col]

        numeric_attempt = pd.to_numeric(series, errors="coerce")

        if numeric_attempt.notna().all():
            # Every value converted cleanly -> truly numeric column
            X[col] = numeric_attempt
        else:
            # Has real text in it (e.g. "Male"/"Female") -> label encode
            le = LabelEncoder()
            X[col] = le.fit_transform(series.astype(str))
            encoders[col] = le

    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    X = X.astype(float)

    return X, encoders


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

            st.subheader("Full dataset")
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

            st.subheader("Missing Values (after cleaning, should be 0)")
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

                # Drop rows where the target itself is missing
                df = df.dropna(subset=[target])

                X = df.drop(columns=[target])
                y = df[target]

                with st.expander("🔍 Data preview before training (debug info)"):
                    st.write("Column data types:")
                    st.write(X.dtypes.astype(str))
                    if X.columns.duplicated().any():
                        st.warning(f"Duplicate column names found and removed: "
                                   f"{list(X.columns[X.columns.duplicated()])}")

                # ---- Encode FEATURES (bulletproof: tries numeric, else label-encodes) ----
                X, encoders = safe_prepare_features(X)

                # ---- Encode TARGET if it's categorical ----
                target_encoder = None
                if y.dtype == "object" or not pd.api.types.is_numeric_dtype(y):
                    target_encoder = LabelEncoder()
                    y = target_encoder.fit_transform(y.astype(str))

                if X.empty or len(X) < 5:
                    st.error("Not enough clean data to train a model. Please check your dataset.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    model = RandomForestClassifier(random_state=42)
                    model.fit(X_train, y_train)

                    pred = model.predict(X_test)

                    acc = accuracy_score(y_test, pred)
                    prec = precision_score(y_test, pred, average="weighted", zero_division=0)
                    rec = recall_score(y_test, pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, pred, average="weighted", zero_division=0)

                    st.success("Model Trained")

                    st.write("Accuracy:", acc)
                    st.write("Precision:", prec)
                    st.write("Recall:", rec)
                    st.write("F1 Score:", f1)

                    joblib.dump(model, "disease_model.pkl")

                    st.session_state.model = model
                    st.session_state.features = list(X.columns)
                    st.session_state.encoders = encoders
                    st.session_state.target_encoder = target_encoder
                    st.session_state.target_name = target

    # ===========================
    # PREDICTION
    # (now correctly nested inside the logged-in block)
    # ===========================

    if menu == "Prediction":

        if st.session_state.model is None:
            st.warning("Train model first")

        else:
            st.subheader("Enter Values")

            inputs = {}
            encoders = st.session_state.encoders or {}

            for col in st.session_state.features:
                if col in encoders:
                    # This was a categorical column during training:
                    # show the original category names, not raw numbers
                    options = list(encoders[col].classes_)
                    choice = st.selectbox(col, options)
                    inputs[col] = encoders[col].transform([choice])[0]
                else:
                    inputs[col] = st.number_input(col, value=0.0)

            if st.button("Predict"):
                input_df = pd.DataFrame([inputs])[st.session_state.features]
                input_df = input_df.astype(float)

                model = st.session_state.model
                prediction = model.predict(input_df)[0]

                # Decode prediction back to the original label if it was encoded
                if st.session_state.target_encoder is not None:
                    prediction_label = st.session_state.target_encoder.inverse_transform([int(prediction)])[0]
                else:
                    prediction_label = prediction

                st.success(f"Predicted Disease: {prediction_label}")

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_df)[0]
                    max_prob = max(probabilities)
                    st.write(f"Prediction Confidence: {max_prob*100:.2f}%")
