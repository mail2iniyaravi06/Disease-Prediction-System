import streamlit as st
import pandas as pd
import joblib
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import base64


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
    with open(image_file, "rb") as image:
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

        /* Optional: make containers transparent */
        .stContainer {{
            background-color: rgba(255,255,255,0.1);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("disease_bg.jpg")

# ===============================
# SESSION STATE INIT
# ===============================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "df" not in st.session_state:
    st.session_state.df = None

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "model" not in st.session_state:
    st.session_state.model = None

if "features" not in st.session_state:
    st.session_state.features = None


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
# LOGIN / REGISTER (CHANGED ORDER)
# ===============================

if not st.session_state.logged_in:

    menu = st.sidebar.radio(
    "Menu",
    ["Register", "Login"]
)

    # REGISTER
    if menu == "Register":
        st.subheader("Create Account")

        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Register"):
            try:
                cursor.execute(
                    "INSERT INTO users(username,password) VALUES (?,?)",
                    (user, pwd)
                )
                conn.commit()
                st.success("Registered Successfully")

            except:
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

st.sidebar.markdown("---")   # line separator

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

            st.subheader("Missing Values")
            st.write(df.isnull().sum())

            df.fillna(df.mode().iloc[0], inplace=True)
            st.success("Missing values handled")

            dup = df.duplicated().sum()
            st.write("Duplicates:", dup)

            df.drop_duplicates(inplace=True)
            st.success("Duplicates removed")

            st.session_state.cleaned_df = df


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

                # -----------------------
                # IMPORTANT FIX HERE
                # Only encode FEATURES, NOT TARGET
                # -----------------------

                X = df.drop(columns=[target])
                y = df[target]

                encoders = {}

                for col in X.columns:   # ❗ only features
                    if X[col].dtype == "object":
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col])
                        encoders[col] = le

                # If target is object, encode separately
                #if y.dtype == "object":
                    #target_le = LabelEncoder()
                    #y = target_le.fit_transform(y)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model = RandomForestClassifier()
                model.fit(X_train, y_train)

                pred = model.predict(X_test)

                acc = accuracy_score(y_test, pred)
                prec = precision_score(y_test, pred, average="weighted")
                rec = recall_score(y_test, pred, average="weighted")
                f1 = f1_score(y_test, pred, average="weighted")

                st.success("Model Trained")

                st.write("Accuracy:", acc)
                st.write("Precision:", prec)
                st.write("Recall:", rec)
                st.write("F1 Score:", f1)

                joblib.dump(model, "disease_model.pkl")

                st.session_state.model = model
                st.session_state.features = list(X.columns)


   # ===========================
# PREDICTION
# ===========================

if menu == "Prediction":

    if st.session_state.model is None:
        st.warning("Train model first")

    else:

        st.subheader("Enter Values")

        inputs = {}

        for col in st.session_state.features:
            inputs[col] = st.number_input(col, value=0.0)

        if st.button("Predict"):

            input_df = pd.DataFrame([inputs])

            model = st.session_state.model

            prediction = model.predict(input_df)[0]

            st.success(f"Predicted Disease: {prediction}")

            # Show only confidence of predicted class
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_df)[0]

                max_prob = max(probabilities)

                st.write(
                    f"Prediction Confidence: {max_prob*100:.2f}%"
                )
                
