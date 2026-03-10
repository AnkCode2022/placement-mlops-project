from kfp import dsl, compiler
from google.cloud import aiplatform
# Import our variables and components
from config import PROJECT_ID, REGION, BUCKET_URI, BQ_TABLE, PIPELINE_ROOT
from components.train import train_placement_model
from components.deploy import deploy_model_to_endpoint

@dsl.pipeline(name="placement-production-pipeline")
def placement_pipeline():
    # Step 1: Train from BigQuery
    train_task = train_placement_model(
        project_id=PROJECT_ID,
        bq_table=BQ_TABLE
    )

    # Step 2: Deploy to Vertex Endpoint
    # We use a pre-built Google container for Scikit-Learn/XGBoost prediction
    deploy_task = deploy_model_to_endpoint(
        project_id=PROJECT_ID,
        region=REGION,
        model_display_name="student-placement-v1",
        serving_container_image="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
        model_artifact=train_task.outputs['model_artifact']
    )

if __name__ == "__main__":
    # Compile the pipeline into a YAML file
    compiler.Compiler().compile(
        pipeline_func=placement_pipeline, 
        package_path="placement_pipeline.yaml"
    )

    # Initialize and Run on Vertex AI
    aiplatform.init(project=PROJECT_ID, location=REGION)
    job = aiplatform.PipelineJob(
        display_name="placement-run-001",
        template_path="placement_pipeline.yaml",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False
    )
    job.submit()