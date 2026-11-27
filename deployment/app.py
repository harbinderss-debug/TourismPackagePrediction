
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ============================================================
# 3. APPLICATION HEADER
# ============================================================

st.title("🌍 Visit With Us")
st.subheader("Tourism Package Purchase Prediction")

st.markdown(
    """
    This application predicts whether a potential customer
    is likely to purchase a tourism package based on their
    demographic, financial and travel-related information.
    """
)

st.divider()

# ============================================================
# 4. COLLECT CUSTOMER INPUTS
# ============================================================

with st.form("customer_prediction_form"):

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # Personal Information
    # --------------------------------------------------------

    with col1:

        st.subheader("Personal Information")

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced", "Unmarried"]
        )

        occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Small Business", "Large Business", "Free Lancer"]
        )

        designation = st.selectbox(
            "Designation",
            [
                "Executive",
                "Manager",
                "Senior Manager",
                "AVP",
                "VP"
            ]
        )

    # --------------------------------------------------------
    # Financial and Location Information
    # --------------------------------------------------------

    with col2:

        st.subheader("Financial Information")

        monthly_income = st.number_input(
            "Monthly Income (INR)",
            min_value=0,
            value=25000,
            step=1000
        )

        city_tier = st.selectbox(
            "City Tier",
            [1, 2, 3]
        )

        own_car = st.selectbox(
            "Owns a Car",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        passport = st.selectbox(
            "Has Passport",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    # --------------------------------------------------------
    # Travel Information
    # --------------------------------------------------------

    with col3:

        st.subheader("Travel Information")

        number_of_trips = st.number_input(
            "Number of Trips",
            min_value=0,
            max_value=30,
            value=3
        )

        number_of_person_visiting = st.number_input(
            "Number of Persons Visiting",
            min_value=1,
            max_value=20,
            value=2
        )

        number_of_children_visiting = st.number_input(
            "Number of Children Visiting",
            min_value=0,
            max_value=19,
            value=0
        )

        preferred_property_star = st.selectbox(
            "Preferred Property Star Rating",
            [3, 4, 5]
        )

    submitted = st.form_submit_button(
        "Predict Purchase Likelihood",
        type="primary",
        use_container_width=True
    )

# ============================================================
# 5. CREATE INPUT DATAFRAME
# ============================================================

if submitted:

    if number_of_children_visiting > number_of_person_visiting:
        st.error(
            "Number of children cannot exceed "
            "the total number of persons visiting."
        )
        st.stop()

    input_data = pd.DataFrame({

        "Age": [age],
        "CityTier": [city_tier],
        "NumberOfPersonVisiting": [number_of_person_visiting],
        "PreferredPropertyStar": [preferred_property_star],
        "NumberOfTrips": [number_of_trips],
        "Passport": [passport],
        "OwnCar": [own_car],
        "NumberOfChildrenVisiting": [number_of_children_visiting],
        "MonthlyIncome": [monthly_income],
        "Occupation": [occupation],
        "Gender": [gender],
        "MaritalStatus": [marital_status],
        "Designation": [designation]

    })

    # Ensure the feature order matches model training
    if hasattr(model, "feature_names_in_"):
        input_data = input_data[list(model.feature_names_in_)]

    # ========================================================
    # 6. MODEL PREDICTION
    # ========================================================

    try:

        prediction = model.predict(input_data)[0]

        positive_class_index = list(model.classes_).index(1)

        probability = model.predict_proba(
            input_data
        )[0][positive_class_index]

        # ====================================================
        # 7. DISPLAY PREDICTION RESULTS
        # ====================================================

        st.divider()
        st.subheader("Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == 1:
                st.success("Customer is likely to purchase the package")
            else:
                st.warning("Customer is unlikely to purchase the package")

        with result_col2:

            st.metric(
                label="Purchase Probability",
                value=f"{probability:.2%}"
            )

            st.progress(float(probability))

        # ====================================================
        # 8. BUSINESS RECOMMENDATION
        # ====================================================

        st.subheader("Business Recommendation")

        if prediction == 1:
            st.info(
                "Prioritize this customer for targeted marketing, "
                "personalized tourism packages and sales follow-up."
            )
        else:
            st.info(
                "Consider lower-cost marketing campaigns, "
                "promotional offers and customer engagement activities."
            )

        # ====================================================
        # 9. DISPLAY INPUT DATAFRAME
        # ====================================================

        with st.expander("View Customer Input Data"):
            st.dataframe(
                input_data,
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ============================================================
# 10. FOOTER
# ============================================================

st.divider()

st.caption(
    "Visit With Us | Tourism Package Prediction | "
    "Machine Learning and MLOps Project"
)

