from kfp import dsl

@dsl.component(packages_to_install=["google-cloud-aiplatform"])
def deploy_model_to_endpoint(
    project_id: str,
    region: str,
    model_display_name: str,
    serving_container_image: str,
    model_artifact: dsl.Input[dsl.Model]
):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    # 1. Upload to Model Registry
    uploaded_model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_artifact.uri,
        serving_container_image_uri=serving_container_image,
    )

    # 2. Deploy to Endpoint
    endpoint = uploaded_model.deploy(
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1
    )