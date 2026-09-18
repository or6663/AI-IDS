from pathlib import Path

import altair as alt
import joblib
import streamlit as st


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="AI Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# Model Configuration
# ============================================================

MODEL_PATH = Path("models/xgboost_ids_final.joblib")


# ============================================================
# Load Frozen IDS Model
# ============================================================

@st.cache_resource
def load_ids_model():
    """
    Load the frozen IDS model bundle created during Day 3.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# Dashboard
# ============================================================

st.title("🛡️ AI Network Intrusion Detection System")

st.write(
    "Machine Learning-Based Network Intrusion Detection Prototype"
)

try:
    ids_pipeline = load_ids_model()

    threshold = ids_pipeline["threshold"]
    feature_columns = ids_pipeline["feature_columns"]
    preprocessor = ids_pipeline["preprocessor"]
    model = ids_pipeline["model"]

    st.success("Frozen IDS model loaded successfully.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Classification Threshold",
            f"{threshold:.2f}"
        )

    with col2:
        st.metric(
            "Required Input Features",
            len(feature_columns)
        )

except Exception as error:
    st.error(f"Failed to load IDS model: {error}")


    # ============================================================
# CSV Upload
# ============================================================

st.divider()

st.header("Network Flow Analysis")

uploaded_file = st.file_uploader(
    "Upload a network flow CSV file",
    type=["csv"],
)

if uploaded_file is not None:

    try:
        import pandas as pd

        input_data = pd.read_csv(uploaded_file)

        st.write(f"Uploaded flows: {len(input_data)}")

        # Check whether all 42 required features exist
        missing_features = [
            feature
            for feature in feature_columns
            if feature not in input_data.columns
        ]

        if missing_features:
            st.error(
                f"Invalid CSV: {len(missing_features)} required "
                "feature(s) are missing."
            )

            st.write("Missing features:")
            st.write(missing_features)

        else:
            st.success(
                "CSV schema validation passed. "
                "All 42 required features are available."
            )

            # Keep only the exact features used by the model
            model_input = input_data[feature_columns].copy()

            st.subheader("Input Data Preview")
            st.dataframe(
                model_input.head(10),
                use_container_width=True,
            )
            # ============================================================
            # IDS Prediction
            # ============================================================

            processed_data = preprocessor.transform(model_input)

            attack_probabilities = model.predict_proba(
                processed_data
            )[:, 1]

            predictions = (
                attack_probabilities >= threshold
            ).astype(int)

            # ============================================================
            # Detection Summary
            # ============================================================

            total_flows = len(predictions)
            normal_flows = int((predictions == 0).sum())
            attack_flows = int((predictions == 1).sum())

            st.subheader("Detection Summary")

            # Calculate attack ratio
            attack_ratio = (
                attack_flows / total_flows * 100
                if total_flows > 0
                else 0
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Flows",
                    total_flows
                )

            with col2:
                st.metric(
                    "Normal Flows",
                    normal_flows
                )

            with col3:
                st.metric(
                    "Attack Flows",
                    attack_flows
                )

            with col4:
                st.metric(
                    "Attack Ratio",
                    f"{attack_ratio:.1f}%"
                )

            # Display security alert
            if attack_flows > 0:
                st.warning(
                    f"⚠️ Security Alert: "
                    f"{attack_flows} suspicious flow(s) detected."
                )
            else:
                st.success(
                    "✅ No suspicious network flows detected."
                )

            # ============================================================
            # Detection Distribution
            # ============================================================

            st.subheader("Detection Distribution")

            distribution_data = pd.DataFrame({
                "Traffic Type": ["Normal", "Attack"],
                "Flows": [normal_flows, attack_flows],
            })


            distribution_data["Traffic Type"] = pd.Categorical(
                distribution_data["Traffic Type"],
                categories=["Normal", "Attack"],
                ordered=True,
            )

            distribution_data = distribution_data.sort_values(
                "Traffic Type"
            )


            chart_col, probability_col = st.columns([1, 1])

            with chart_col:
                st.bar_chart(
                    distribution_data,
                    x="Traffic Type",
                    y="Flows",
                    height=300,
                )

            with probability_col:
                st.markdown("#### Attack Probability Overview")

                probability_data = pd.DataFrame({
                    "Flow": [
                        f"Flow {i}"
                        for i in range(1, len(attack_probabilities) + 1)
                    ],
                    "Attack Probability": attack_probabilities,
                })

                probability_chart = alt.Chart(
                    probability_data
                ).mark_bar().encode(
                    x=alt.X(
                        "Flow:N",
                        sort=None,
                        title="Flow"
                    ),
                    y=alt.Y(
                        "Attack Probability:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        title="Attack Probability"
                    ),
                    tooltip=[
                        "Flow:N",
                        alt.Tooltip(
                            "Attack Probability:Q",
                            format=".4f"
                        ),
                    ],
                )

                threshold_line = alt.Chart(
                    pd.DataFrame({
                        "Threshold": [threshold]
                    })
                ).mark_rule(
                    strokeDash=[6, 4]
                ).encode(
                    y="Threshold:Q"
                )

                threshold_label = alt.Chart(
                    pd.DataFrame({
                        "Threshold": [threshold],
                        "Label": [f"Threshold = {threshold:.2f}"]
                    })
                ).mark_text(
                    align="left",
                    baseline="bottom",
                    dx=5,
                    dy=-3
                ).encode(
                    y="Threshold:Q",
                    text="Label:N"
                )


                final_probability_chart = (
                    probability_chart
                    + threshold_line
                    + threshold_label
                ).properties(
                    height=300
                )

                st.altair_chart(
                    final_probability_chart,
                    use_container_width=True
                )


            prediction_results = pd.DataFrame({
                "Flow": range(1, len(model_input) + 1),
                "Attack Probability": attack_probabilities,
                "Prediction": [
                    "⚠️ ATTACK" if prediction == 1 else "✅ NORMAL"
                    for prediction in predictions
                ],
            })

            prediction_results["Attack Probability"] = (
                prediction_results["Attack Probability"].round(4)
            )

            st.subheader("Detection Results")

            st.dataframe(
                prediction_results,
                use_container_width=True,
                hide_index=True,
            )


            # ============================================================
            # Export Detection Results
            # ============================================================

            export_data = model_input.copy()

            export_data["Attack Probability"] = attack_probabilities

            export_data["Prediction"] = [
                "ATTACK" if prediction == 1 else "NORMAL"
                for prediction in predictions
            ]

            csv_output = export_data.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Detection Results",
                data=csv_output,
                file_name="ids_detection_results.csv",
                mime="text/csv",
            )


    except Exception as error:
        st.error(f"Failed to read CSV: {error}")