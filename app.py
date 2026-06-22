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
    "model": None,
    "features": None,
    "encoders": {},        # col -> LabelEncoder  (only STRING feature columns)
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
        pwd  = st.text_input("Password", type="password")
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
        pwd  = st.text_input("Password", type="password")
        if st.button("Login"):
            cursor.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (user, pwd)
            )
            result = cursor.fetchone()
            if result:
                st.session_state.logged_in = True
                st.session_state.username  = user
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
    """Fill missing values; numeric → median, string → mode."""
    df = df.copy()

    # Try converting object columns that are secretly numeric
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")
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
                fill_val  = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
                df[col]   = df[col].fillna(fill_val)
            df[col] = df[col].astype(str)

    return df


# ===============================
# LABEL ENCODE ONLY STRING FEATURE COLUMNS
# Numeric columns are left as-is.
# Target column is NEVER touched here.
# ===============================
def label_encode_string_features(X: pd.DataFrame):
    """
    - Detects every column that contains string / object data.
    - Applies LabelEncoder to convert each string column → 0/1/2… integers.
    - Numeric columns are kept exactly as they are.
    - Returns the transformed X and a dict of {col: fitted LabelEncoder}.
    """
    X = X.copy()

    # Remove duplicate column names silently
    if X.columns.duplicated().any():
        X = X.loc[:, ~X.columns.duplicated()]

    encoders = {}

    for col in X.columns:
        # Check if this column has string/object values
        if X[col].dtype == "object" or (
            X[col].dtype != "object" and
            not pd.api.types.is_numeric_dtype(X[col])
        ):
            le = LabelEncoder()
            X[col]       = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        else:
            # Numeric column — just make sure it has no NaN / inf
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
            X[col] = X[col].replace([np.inf, -np.inf], 0)

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

                # Separate features and target BEFORE any encoding
                X = df.drop(columns=[target])
                y = df[target]   # target is NEVER label-encoded

                # -------------------------------------------------------
                # LABEL ENCODE: only string columns in X, skip numeric ones
                # -------------------------------------------------------
                X_encoded, encoders = label_encode_string_features(X)

                # Show which string columns were encoded
                if encoders:
                    with st.expander("📊 Label Encoding Summary (string feature columns only)"):
                        for col, le in encoders.items():
                            mapping = {str(orig): int(enc) for enc, orig in enumerate(le.classes_)}
                            st.write(f"**{col}** → {mapping}")
                else:
                    st.info("No string columns found in features — all columns are already numeric.")

                # Show numeric columns that were kept as-is
                numeric_cols = [c for c in X.columns if c not in encoders]
                if numeric_cols:
                    st.info(f"Numeric columns (kept as-is): {numeric_cols}")

                # -------------------------------------------------------
                # TARGET: keep original values — NO encoding
                # -------------------------------------------------------
                # y stays exactly as it is (string or numeric)
                target_encoder = None
                if not pd.api.types.is_numeric_dtype(y):
                    # We still need to encode for sklearn, but we store the encoder
                    # so we can decode predictions back to original labels
                    target_encoder = LabelEncoder()
                    y_encoded = target_encoder.fit_transform(y.astype(str))
                    st.info(
                        f"Target column **'{target}'** contains strings — "
                        f"encoded internally for training only. "
                        f"Mapping: { {str(orig): int(enc) for enc, orig in enumerate(target_encoder.classes_)} }"
                    )
                else:
                    y_encoded = y.values

                if X_encoded.empty or len(X_encoded) < 5:
                    st.error("Not enough data to train. Please check your dataset.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_encoded, y_encoded, test_size=0.2, random_state=42
                    )

                    model = RandomForestClassifier(random_state=42)
                    model.fit(X_train, y_train)

                    pred = model.predict(X_test)

                    acc  = accuracy_score(y_test, pred)
                    prec = precision_score(y_test, pred, average="weighted", zero_division=0)
                    rec  = recall_score(y_test, pred, average="weighted", zero_division=0)
                    f1   = f1_score(y_test, pred, average="weighted", zero_division=0)

                    st.success("✅ Model Trained Successfully!")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy",  f"{acc:.4f}")
                    c2.metric("Precision", f"{prec:.4f}")
                    c3.metric("Recall",    f"{rec:.4f}")
                    c4.metric("F1 Score",  f"{f1:.4f}")

                    joblib.dump(model, "disease_model.pkl")

                    st.session_state.model          = model
                    st.session_state.features       = list(X_encoded.columns)
                    st.session_state.encoders       = encoders          # only string cols
                    st.session_state.target_encoder = target_encoder    # None if target is numeric
                    st.session_state.target_name    = target

    # ===========================
    # PREDICTION
    # ===========================
    if menu == "Prediction":

        if st.session_state.model is None:
            st.warning("Train model first")

        else:
            st.subheader("Enter Values for Prediction")

            inputs   = {}
            encoders = st.session_state.encoders or {}

            for col in st.session_state.features:
                if col in encoders:
                    # String column — show original category names as dropdown
                    le      = encoders[col]
                    options = list(le.classes_)
                    choice  = st.selectbox(col, options)
                    inputs[col] = int(le.transform([choice])[0])
                else:
                    # Numeric column — show number input
                    inputs[col] = st.number_input(col, value=0.0)

            if st.button("Predict"):
                input_df   = pd.DataFrame([inputs])[st.session_state.features]
                input_df   = input_df.astype(float)

                model      = st.session_state.model
                prediction = model.predict(input_df)[0]

                # Decode prediction back to original target label
                if st.session_state.target_encoder is not None:
                    prediction_label = st.session_state.target_encoder.inverse_transform(
                        [int(prediction)]
                    )[0]
                else:
                    prediction_label = prediction

                st.success(f"🩺 Predicted Disease: **{prediction_label}**")

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_df)[0]
                    max_prob      = max(probabilities)
                    st.write(f"🎯 Prediction Confidence: **{max_prob * 100:.2f}%**")

                    with st.expander("📈 All Class Probabilities"):
                        classes    = model.classes_
                        target_enc = st.session_state.target_encoder
                        prob_rows  = []
                        for cls, prob in zip(classes, probabilities):
                            if target_enc is not None:
                                label = target_enc.inverse_transform([int(cls)])[0]
                            else:
                                label = cls
                            prob_rows.append({"Disease": str(label),
                                              "Probability (%)": round(float(prob) * 100, 2)})
                        prob_df = (
                            pd.DataFrame(prob_rows)
                            .sort_values("Probability (%)", ascending=False)
                            .reset_index(drop=True)
                        )
                        st.dataframe(prob_df, use_container_width=True)
