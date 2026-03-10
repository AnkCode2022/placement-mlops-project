from kfp import dsl

@dsl.component(
    packages_to_install=["pandas", "google-cloud-bigquery", "xgboost", "pyarrow, "db-dtypes"],
    base_image="python:3.10"
)
def train_placement_model(
    project_id: str,
    bq_table: str,
    model_artifact: dsl.Output[dsl.Model]
):
    from google.cloud import bigquery
    import pandas as pd
    from xgboost import XGBClassifier
    import pickle
    import os

    # 1. Fetch data from BigQuery
    client = bigquery.Client(project=project_id)
    query = f"SELECT cgpa, iq, profile_score, placed FROM `{bq_table}`"
    df = client.query(query).to_dataframe()

    # 2. Train Model
    X = df[['cgpa', 'iq', 'profile_score']]
    y = df['placed']
    model = XGBClassifier()
    model.fit(X, y)

    # 3. Save model to the path Vertex AI expects (GCS)
    os.makedirs(model_artifact.path, exist_ok=True)
    with open(os.path.join(model_artifact.path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    
    model_artifact.metadata["framework"] = "XGBoost"