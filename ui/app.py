from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "heart_disease_features_Raw.csv"
TARGET_PATH = PROJECT_ROOT / "data" / "heart_disease_target_Raw.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model_pipeline.pkl"

NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


class RoundCategoricalInts(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns if len(self.columns) == X.shape[1] else None)
        for column in self.columns:
            if column in X.columns:
                X[column] = X[column].round().astype(int)
        return X.values


class OutlierCapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        for column in X.columns:
            first_quartile = X[column].quantile(0.25)
            third_quartile = X[column].quantile(0.75)
            interquartile_range = third_quartile - first_quartile
            lower = first_quartile - 1.5 * interquartile_range
            upper = third_quartile + 1.5 * interquartile_range
            X[column] = np.clip(X[column], lower, upper)
        return X.values


# The saved notebook pipeline refers to these transformer classes as __main__.
setattr(sys.modules["__main__"], "RoundCategoricalInts", RoundCategoricalInts)
setattr(sys.modules["__main__"], "OutlierCapper", OutlierCapper)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_training_data():
    if not FEATURES_PATH.is_file() or not TARGET_PATH.is_file():
        raise FileNotFoundError("Raw feature and target CSV files are required in the data folder.")

    features = pd.read_csv(FEATURES_PATH)
    target = pd.read_csv(TARGET_PATH).iloc[:, 0]
    if len(features) != len(target):
        raise ValueError("The feature and target CSV files have different row counts.")
    if not set(FEATURE_COLUMNS).issubset(features.columns):
        raise ValueError("The raw feature CSV does not contain the columns expected by the model.")

    data = features.copy()
    data["target"] = (target > 0).astype(int).to_numpy()
    return data


def render_prediction_form():
    with st.form("heart_data_form"):
        st.subheader("Patient measurements")
        st.caption("Enter values from a clinical assessment. No information is saved by this app.")

        first_column, second_column = st.columns(2)
        with first_column:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=54)
            sex = st.selectbox("Sex in source data", options=[0, 1], format_func=lambda value: {
                0: "Female (0)", 1: "Male (1)"
            }[value])
            chest_pain = st.selectbox(
                "Chest pain type",
                options=[1, 2, 3, 4],
                format_func=lambda value: {
                    1: "Typical angina (1)",
                    2: "Atypical angina (2)",
                    3: "Non-anginal pain (3)",
                    4: "Asymptomatic (4)",
                }[value],
            )
            resting_bp = st.number_input("Resting blood pressure (mm Hg)", min_value=50, max_value=300, value=130)
            cholesterol = st.number_input("Serum cholesterol (mg/dL)", min_value=50, max_value=700, value=240)
            fasting_blood_sugar = st.selectbox("Fasting blood sugar > 120 mg/dL", options=[0, 1], format_func=lambda value: "Yes" if value else "No")
            resting_ecg = st.selectbox(
                "Resting ECG",
                options=[0, 1, 2],
                format_func=lambda value: {
                    0: "Normal (0)",
                    1: "ST-T wave abnormality (1)",
                    2: "Left ventricular hypertrophy (2)",
                }[value],
            )
        with second_column:
            max_heart_rate = st.number_input("Maximum heart rate achieved", min_value=40, max_value=250, value=150)
            exercise_angina = st.selectbox("Exercise-induced angina", options=[0, 1], format_func=lambda value: "Yes" if value else "No")
            st_depression = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Peak exercise ST slope", options=[1, 2, 3], format_func=lambda value: {
                1: "Upsloping (1)", 2: "Flat (2)", 3: "Downsloping (3)"
            }[value])
            major_vessels = st.selectbox("Major vessels colored by fluoroscopy (ca)", options=[0, 1, 2, 3])
            thal = st.selectbox("Thalassemia test (thal)", options=[3, 6, 7], format_func=lambda value: {
                3: "Normal (3)", 6: "Fixed defect (6)", 7: "Reversible defect (7)"
            }[value])

        submitted = st.form_submit_button("Estimate model output", type="primary")

    if submitted:
        patient = pd.DataFrame(
            [[age, resting_bp, cholesterol, max_heart_rate, st_depression, major_vessels,
              sex, chest_pain, fasting_blood_sugar, resting_ecg, exercise_angina, slope, thal]],
            columns=FEATURE_COLUMNS,
        )
        model = load_model()
        prediction = int(model.predict(patient)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            class_labels = list(model.classes_)
            if 1 in class_labels:
                probability = float(model.predict_proba(patient)[0][class_labels.index(1)])

        st.subheader("Model result")
        if prediction == 1:
            st.warning("The model classified this input in the heart-disease-positive class.")
        else:
            st.success("The model classified this input in the heart-disease-negative class.")
        if probability is not None:
            st.metric("Model score for the positive class", f"{probability:.1%}")
        st.caption(
            "This is a machine-learning classification from a small historical dataset, "
            "not a diagnosis or a clinically validated estimate of personal risk."
        )


def render_data_exploration(data):
    st.header("Explore the project dataset")
    st.caption(
        "Use the filters to compare cohorts. Charts support hover, zoom, and drag selection. "
        "The target is binarized as 0 = source value 0 and 1 = source value above 0."
    )

    with st.expander("Filter the cohort", expanded=True):
        age_limits = (int(data["age"].min()), int(data["age"].max()))
        age_range = st.slider(
            "Age range",
            min_value=age_limits[0],
            max_value=age_limits[1],
            value=age_limits,
            key="eda_age_range",
        )
        filter_columns = st.columns(3)
        with filter_columns[0]:
            labels = st.multiselect(
                "Dataset label",
                options=[0, 1],
                default=[0, 1],
                format_func=lambda value: "Disease-positive (1)" if value else "Disease-negative (0)",
                key="eda_labels",
            )
        with filter_columns[1]:
            sex_values = sorted(data["sex"].dropna().unique().tolist())
            selected_sex = st.multiselect(
                "Sex",
                options=sex_values,
                default=sex_values,
                format_func=lambda value: {0: "Female (0)", 1: "Male (1)"}.get(value, str(value)),
                key="eda_sex",
            )
        with filter_columns[2]:
            pain_values = sorted(data["cp"].dropna().unique().tolist())
            selected_pain = st.multiselect(
                "Chest pain type",
                options=pain_values,
                default=pain_values,
                format_func=lambda value: {
                    1: "Typical angina (1)",
                    2: "Atypical angina (2)",
                    3: "Non-anginal pain (3)",
                    4: "Asymptomatic (4)",
                }.get(value, str(value)),
                key="eda_chest_pain",
            )

    cohort = data.loc[
        data["age"].between(*age_range)
        & data["target"].isin(labels)
        & data["sex"].isin(selected_sex)
        & data["cp"].isin(selected_pain)
    ].copy()

    if cohort.empty:
        st.warning("No records match these filters. Widen the age range or select more categories.")
        return

    cohort["Disease label"] = cohort["target"].map({0: "Negative", 1: "Positive"})
    full_positive_rate = data["target"].mean()
    cohort_positive_rate = cohort["target"].mean()
    metric_columns = st.columns(4)
    metric_columns[0].metric("Filtered records", f"{len(cohort):,}", f"of {len(data):,}")
    metric_columns[1].metric("Positive labels", f"{cohort_positive_rate:.1%}")
    metric_columns[2].metric("Dataset baseline", f"{full_positive_rate:.1%}")
    missing_count = int(cohort[FEATURE_COLUMNS].isna().sum().sum())
    metric_columns[3].metric("Missing feature values", f"{missing_count:,}")

    overview_tab, distributions_tab, relationships_tab, patterns_tab = st.tabs(
        ["Overview", "Distributions", "Relationships", "PCA & clusters"]
    )

    with overview_tab:
        st.subheader("Cohort snapshot")
        overview_columns = st.columns(2)
        with overview_columns[0]:
            label_counts = (
                cohort["Disease label"]
                .value_counts()
                .rename_axis("Dataset label")
                .reset_index(name="Records")
            )
            figure = px.bar(
                label_counts,
                x="Dataset label",
                y="Records",
                color="Dataset label",
                color_discrete_map={"Negative": "#3b82f6", "Positive": "#ef4444"},
                title="Class balance in the filtered cohort",
            )
            st.plotly_chart(figure, width="stretch")
        with overview_columns[1]:
            missing = cohort[FEATURE_COLUMNS].isna().sum().rename("Missing values")
            missing = missing[missing > 0].rename_axis("Feature").reset_index()
            if missing.empty:
                st.success("No missing values in this filtered cohort.")
            else:
                figure = px.bar(
                    missing,
                    x="Feature",
                    y="Missing values",
                    color="Feature",
                    title="Missing values to consider during preprocessing",
                )
                st.plotly_chart(figure, width="stretch")

        st.subheader("Descriptive statistics")
        numeric_summary = cohort[NUMERIC_COLUMNS].describe().T
        st.dataframe(numeric_summary.style.format("{:.2f}"), width="stretch")

        st.subheader("Records in this cohort")
        preview = cohort[FEATURE_COLUMNS + ["target"]].copy()
        preview["target"] = preview["target"].map({0: "Negative", 1: "Positive"})
        st.dataframe(preview, width="stretch", height=280)

        if len(cohort) >= 2:
            rate_difference = cohort_positive_rate - full_positive_rate
            comparison = "higher" if rate_difference > 0 else "lower" if rate_difference < 0 else "the same as"
            st.info(
                f"This cohort contains {len(cohort)} of {len(data)} records. Its positive-label share "
                f"({cohort_positive_rate:.1%}) is {comparison} the full dataset share ({full_positive_rate:.1%}). "
                "This is a descriptive comparison, not a clinical risk estimate."
            )

    with distributions_tab:
        st.subheader("Inspect one feature at a time")
        distribution_kind = st.radio(
            "Feature type",
            options=["Numeric", "Categorical"],
            horizontal=True,
            key="eda_distribution_kind",
        )
        if distribution_kind == "Numeric":
            feature_options = NUMERIC_COLUMNS
        else:
            feature_options = CATEGORICAL_COLUMNS + ["ca"]
        selected_feature = st.selectbox(
            "Feature",
            options=feature_options,
            format_func=lambda value: {
                "trestbps": "Resting blood pressure (trestbps)",
                "chol": "Cholesterol (chol)",
                "thalach": "Maximum heart rate (thalach)",
                "oldpeak": "ST depression (oldpeak)",
                "ca": "Major vessels (ca)",
            }.get(value, value),
            key="eda_distribution_feature",
        )

        if distribution_kind == "Numeric":
            figure = px.histogram(
                cohort,
                x=selected_feature,
                color="Disease label",
                marginal="box",
                barmode="overlay",
                opacity=0.72,
                nbins=25,
                color_discrete_map={"Negative": "#3b82f6", "Positive": "#ef4444"},
                title=f"{selected_feature} distribution by dataset label",
            )
            st.plotly_chart(figure, width="stretch")
            box_figure = px.box(
                cohort,
                x="Disease label",
                y=selected_feature,
                color="Disease label",
                points="all",
                color_discrete_map={"Negative": "#3b82f6", "Positive": "#ef4444"},
                title=f"{selected_feature}: spread by dataset label",
            )
            st.plotly_chart(box_figure, width="stretch")
            medians = cohort.groupby("Disease label", observed=True)[selected_feature].median()
            if len(medians) == 2:
                st.caption(
                    f"Median {selected_feature}: negative {medians.get('Negative', float('nan')):.2f}; "
                    f"positive {medians.get('Positive', float('nan')):.2f}. "
                    "Differences describe this sample and do not establish cause."
                )
        else:
            category_data = cohort.dropna(subset=[selected_feature]).copy()
            category_data["Category"] = category_data[selected_feature].map(str)
            figure = px.histogram(
                category_data,
                x="Category",
                color="Disease label",
                barmode="group",
                color_discrete_map={"Negative": "#3b82f6", "Positive": "#ef4444"},
                title=f"{selected_feature} counts by dataset label",
                text_auto=True,
            )
            st.plotly_chart(figure, width="stretch")
            st.caption(
                f"{int(cohort[selected_feature].isna().sum())} records in this cohort have no "
                f"{selected_feature} value."
            )

    with relationships_tab:
        st.subheader("Compare two measurements")
        relationship_columns = st.columns(3)
        with relationship_columns[0]:
            x_feature = st.selectbox(
                "Horizontal axis",
                options=NUMERIC_COLUMNS,
                index=0,
                key="eda_x_feature",
            )
        y_options = [column for column in NUMERIC_COLUMNS if column != x_feature]
        with relationship_columns[1]:
            y_feature = st.selectbox(
                "Vertical axis",
                options=y_options,
                index=0,
                key="eda_y_feature",
            )
        with relationship_columns[2]:
            color_choice = st.selectbox(
                "Color points by",
                options=["Disease label", "Sex", "Chest pain type"],
                key="eda_scatter_color",
            )

        scatter_data = cohort.copy()
        if color_choice == "Sex":
            scatter_data["Point group"] = scatter_data["sex"].map(
                {0: "Female", 1: "Male"}
            ).fillna("Unknown")
        elif color_choice == "Chest pain type":
            scatter_data["Point group"] = scatter_data["cp"].map(
                {
                    1: "Typical angina",
                    2: "Atypical angina",
                    3: "Non-anginal pain",
                    4: "Asymptomatic",
                }
            )
        else:
            scatter_data["Point group"] = scatter_data["Disease label"]

        figure = px.scatter(
            scatter_data,
            x=x_feature,
            y=y_feature,
            color="Point group",
            hover_data=["age", "sex", "cp", "target"],
            title=f"{y_feature} compared with {x_feature}",
        )
        st.plotly_chart(figure, width="stretch")

        pair_correlation = cohort[[x_feature, y_feature]].corr().iloc[0, 1]
        if pd.notna(pair_correlation):
            st.caption(
                f"Pearson correlation in the filtered cohort: {pair_correlation:.2f}. "
                "Correlation summarizes a linear association; it does not imply causation."
            )

        st.subheader("Correlation heatmap")
        encoded_features = pd.get_dummies(
            data[FEATURE_COLUMNS],
            columns=CATEGORICAL_COLUMNS,
            prefix=CATEGORICAL_COLUMNS,
            dummy_na=True,
            dtype=float,
        )
        encoded_cohort = encoded_features.loc[cohort.index].copy()
        encoded_cohort["target"] = cohort["target"]
        correlation_matrix = encoded_cohort.corr()
        target_correlations = correlation_matrix["target"].drop("target").dropna()
        top_features = (
            target_correlations.abs().sort_values(ascending=False).head(10).index.tolist()
        )
        selected_correlations = st.multiselect(
            "Features to include",
            options=encoded_features.columns.tolist(),
            default=top_features,
            key="eda_correlation_features",
        )
        if selected_correlations:
            columns = selected_correlations + ["target"]
            heatmap_data = correlation_matrix.loc[columns, columns].fillna(0)
            figure = px.imshow(
                heatmap_data,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                text_auto=".2f",
                aspect="auto",
                title="Feature correlations in the filtered cohort",
            )
            st.plotly_chart(figure, width="stretch")
            if not target_correlations.empty:
                strongest = target_correlations.abs().idxmax()
                st.caption(
                    f"Strongest absolute encoded-feature correlation with the label in this cohort: "
                    f"{strongest} ({target_correlations[strongest]:.2f}). "
                    "This association is descriptive, not causal."
                )
        else:
            st.info("Select at least one feature to display the correlation heatmap.")

    with patterns_tab:
        st.subheader("Principal component view")
        st.caption(
            "Categorical features are one-hot encoded, missing values are median-imputed, "
            "and all columns are standardized for this exploratory analysis."
        )
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import adjusted_rand_score, silhouette_score
        from sklearn.preprocessing import StandardScaler

        if len(cohort) < 3:
            st.info("Select at least three records to explore PCA and clustering.")
            return

        clustering_features = encoded_features.loc[cohort.index]
        imputed_features = SimpleImputer(strategy="median").fit_transform(clustering_features)
        scaled_features = StandardScaler().fit_transform(imputed_features)
        component_count = min(10, len(cohort) - 1, scaled_features.shape[1])
        pca = PCA(n_components=component_count, random_state=42)
        components = pca.fit_transform(scaled_features)
        component_labels = [f"PC{number}" for number in range(1, component_count + 1)]

        plot_controls = st.columns(2)
        with plot_controls[0]:
            x_component = st.selectbox(
                "Horizontal component",
                options=component_labels,
                index=0,
                key="eda_pca_x",
            )
        with plot_controls[1]:
            y_component = st.selectbox(
                "Vertical component",
                options=component_labels,
                index=min(1, component_count - 1),
                key="eda_pca_y",
            )

        pca_data = pd.DataFrame(components, columns=component_labels, index=cohort.index)
        pca_data["Disease label"] = cohort["Disease label"]
        figure = px.scatter(
            pca_data,
            x=x_component,
            y=y_component,
            color="Disease label",
            hover_data=["Disease label"],
            color_discrete_map={"Negative": "#3b82f6", "Positive": "#ef4444"},
            title="PCA projection colored by the dataset label",
        )
        st.plotly_chart(figure, width="stretch")

        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        variance_figure = go.Figure()
        variance_figure.add_trace(
            go.Scatter(
                x=component_labels,
                y=cumulative_variance,
                mode="lines+markers",
                name="Cumulative variance",
            )
        )
        variance_figure.add_hline(y=0.95, line_dash="dash", line_color="#ef4444")
        variance_figure.update_layout(
            title="Cumulative variance retained",
            xaxis_title="Principal component",
            yaxis_title="Cumulative explained variance",
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(variance_figure, width="stretch")

        st.subheader("Unsupervised cluster exploration")
        max_clusters = min(8, len(cohort) - 1)
        cluster_count = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=max_clusters,
            value=min(2, max_clusters),
            key="eda_cluster_count",
        )
        cluster_method = st.radio(
            "Clustering method",
            options=["K-Means", "Hierarchical"],
            horizontal=True,
            key="eda_cluster_method",
        )
        if cluster_method == "K-Means":
            cluster_labels = KMeans(
                n_clusters=cluster_count,
                n_init=10,
                random_state=42,
            ).fit_predict(scaled_features)
        else:
            cluster_labels = AgglomerativeClustering(
                n_clusters=cluster_count,
                linkage="average",
            ).fit_predict(scaled_features)

        cluster_data = pca_data.copy()
        cluster_data["Cluster"] = [f"Cluster {label + 1}" for label in cluster_labels]
        figure = px.scatter(
            cluster_data,
            x=x_component,
            y=y_component,
            color="Cluster",
            symbol="Disease label",
            hover_data=["Disease label"],
            title=f"{cluster_method} clusters in the PCA projection",
        )
        st.plotly_chart(figure, width="stretch")

        cluster_metrics = st.columns(2)
        cluster_metrics[0].metric(
            "Adjusted Rand index vs label",
            f"{adjusted_rand_score(cohort['target'], cluster_labels):.3f}",
        )
        unique_cluster_count = len(np.unique(cluster_labels))
        if 1 < unique_cluster_count < len(cohort):
            cluster_metrics[1].metric(
                "Silhouette score",
                f"{silhouette_score(scaled_features, cluster_labels):.3f}",
            )
        else:
            cluster_metrics[1].metric("Silhouette score", "Unavailable")
        st.caption(
            "Cluster IDs are arbitrary groupings, not disease diagnoses. The scores compare "
            "unsupervised groups with dataset labels and should not be treated as clinical performance."
        )

        if cluster_method == "K-Means":
            elbow_max = min(8, len(cohort) - 1)
            inertias = []
            cluster_range = range(1, elbow_max + 1)
            for candidate_count in cluster_range:
                candidate = KMeans(
                    n_clusters=candidate_count,
                    n_init=10,
                    random_state=42,
                )
                candidate.fit(scaled_features)
                inertias.append(candidate.inertia_)
            elbow_figure = px.line(
                x=list(cluster_range),
                y=inertias,
                markers=True,
                labels={"x": "Number of clusters", "y": "Within-cluster sum of squares"},
                title="K-Means elbow curve",
            )
            st.plotly_chart(elbow_figure, width="stretch")

        cluster_rates = (
            pd.DataFrame({"Cluster": cluster_data["Cluster"], "target": cohort["target"]})
            .groupby("Cluster", observed=True)["target"]
            .mean()
            .rename("Positive label share")
            .reset_index()
        )
        figure = px.bar(
            cluster_rates,
            x="Cluster",
            y="Positive label share",
            range_y=[0, 1],
            title="Positive-label share within each cluster",
        )
        st.plotly_chart(figure, width="stretch")

    st.caption(
        "All plots and statistics describe this dataset or the selected subset only. "
        "Associations, PCA projections, and cluster assignments are not clinical findings."
    )


def main():
    st.set_page_config(page_title="Heart Disease Explorer", page_icon="❤️", layout="wide")
    st.title("Heart Disease Explorer")
    st.write("Explore the UCI Cleveland heart disease dataset and try the project's trained classification pipeline.")
    st.info(
        "For education and demonstration only. This app is not a medical device and cannot diagnose, "
        "rule out, or predict an individual's clinical outcome. Seek advice from a qualified clinician."
    )

    prediction_tab, exploration_tab = st.tabs(["Try a prediction", "Explore the data"])
    with prediction_tab:
        render_prediction_form()
    with exploration_tab:
        render_data_exploration(load_training_data())


if __name__ == "__main__":
    main()
