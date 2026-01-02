"""MLflow Model Registry utilities for model versioning and deployment."""

from typing import Any, List, Optional

import mlflow
from mlflow.tracking import MlflowClient


class ModelRegistry:
    """
    MLflow Model Registry wrapper for managing model versions and stages.

    Provides functionality to:
    - Register models
    - Transition models between stages (Staging, Production, Archived)
    - Load models by name and stage
    - Query model versions
    """

    # Valid stages for model versions
    STAGES = ["None", "Staging", "Production", "Archived"]

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        """
        Initialize the Model Registry.

        Args:
            tracking_uri: MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[dict] = None,
        description: Optional[str] = None,
    ) -> str:
        """
        Register a model from a run artifact.

        Args:
            model_uri: URI of the model artifact (e.g., "runs:/<run_id>/model")
            name: Name to register the model under
            tags: Optional tags for the model version
            description: Optional description

        Returns:
            The version number of the registered model
        """
        result = mlflow.register_model(model_uri, name)

        # Add tags and description if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(name, result.version, key, value)

        if description:
            self.client.update_model_version(
                name=name, version=result.version, description=description
            )

        return result.version

    def transition_model_version_stage(
        self,
        name: str,
        version: int,
        stage: str,
        archive_existing_versions: bool = False,
    ) -> None:
        """
        Transition a model version to a new stage.

        Args:
            name: Registered model name
            version: Model version number
            stage: Target stage ("Staging", "Production", or "Archived")
            archive_existing_versions: Whether to archive existing versions in the target stage
        """
        if stage not in self.STAGES:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {self.STAGES}")

        self.client.transition_model_version_stage(
            name=name,
            version=str(version),
            stage=stage,
            archive_existing_versions=archive_existing_versions,
        )

    def promote_to_staging(self, name: str, version: int) -> None:
        """
        Promote a model version to Staging.

        Args:
            name: Registered model name
            version: Model version number
        """
        self.transition_model_version_stage(name, version, "Staging")

    def promote_to_production(
        self, name: str, version: int, archive_existing: bool = True
    ) -> None:
        """
        Promote a model version to Production.

        Args:
            name: Registered model name
            version: Model version number
            archive_existing: Whether to archive existing production versions
        """
        self.transition_model_version_stage(
            name, version, "Production", archive_existing_versions=archive_existing
        )

    def archive_model_version(self, name: str, version: int) -> None:
        """
        Archive a model version.

        Args:
            name: Registered model name
            version: Model version number
        """
        self.transition_model_version_stage(name, version, "Archived")

    def load_model(self, name: str, version: Optional[int] = None, stage: Optional[str] = None) -> Any:
        """
        Load a registered model.

        Args:
            name: Registered model name
            version: Specific version to load (mutually exclusive with stage)
            stage: Stage to load from (mutually exclusive with version)

        Returns:
            The loaded model

        Raises:
            ValueError: If neither version nor stage is specified, or both are specified
        """
        if version is not None and stage is not None:
            raise ValueError("Specify either version or stage, not both")

        if version is not None:
            model_uri = f"models:/{name}/{version}"
        elif stage is not None:
            if stage not in self.STAGES:
                raise ValueError(f"Invalid stage: {stage}. Must be one of {self.STAGES}")
            model_uri = f"models:/{name}/{stage}"
        else:
            raise ValueError("Must specify either version or stage")

        return mlflow.pytorch.load_model(model_uri)

    def load_production_model(self, name: str) -> Any:
        """
        Load the production version of a model.

        Args:
            name: Registered model name

        Returns:
            The production model
        """
        return self.load_model(name, stage="Production")

    def load_staging_model(self, name: str) -> Any:
        """
        Load the staging version of a model.

        Args:
            name: Registered model name

        Returns:
            The staging model
        """
        return self.load_model(name, stage="Staging")

    def get_latest_versions(
        self, name: str, stages: Optional[List[str]] = None
    ) -> List[Any]:
        """
        Get the latest versions of a model, optionally filtered by stage.

        Args:
            name: Registered model name
            stages: List of stages to filter by

        Returns:
            List of model version objects
        """
        return self.client.get_latest_versions(name, stages=stages)

    def get_latest_version(
        self, name: str, stage: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get the latest version of a model.

        Args:
            name: Registered model name
            stage: Optional stage to filter by

        Returns:
            The latest model version or None if not found
        """
        stages = [stage] if stage else None
        versions = self.get_latest_versions(name, stages=stages)
        return versions[0] if versions else None

    def get_model_version(self, name: str, version: int) -> Any:
        """
        Get a specific model version.

        Args:
            name: Registered model name
            version: Version number

        Returns:
            The model version object
        """
        return self.client.get_model_version(name, str(version))

    def search_model_versions(
        self, filter_string: str = "", max_results: int = 100
    ) -> List[Any]:
        """
        Search for model versions.

        Args:
            filter_string: Filter expression (e.g., "name='my-model'")
            max_results: Maximum number of results

        Returns:
            List of matching model versions
        """
        return self.client.search_model_versions(
            filter_string=filter_string, max_results=max_results
        )

    def get_registered_model(self, name: str) -> Optional[Any]:
        """
        Get a registered model by name.

        Args:
            name: Model name

        Returns:
            The registered model or None if not found
        """
        try:
            return self.client.get_registered_model(name)
        except Exception:
            return None

    def list_registered_models(self, max_results: int = 100) -> List[Any]:
        """
        List all registered models.

        Args:
            max_results: Maximum number of results

        Returns:
            List of registered models
        """
        return list(self.client.search_registered_models(max_results=max_results))

    def delete_model_version(self, name: str, version: int) -> None:
        """
        Delete a model version.

        Args:
            name: Registered model name
            version: Version number to delete
        """
        self.client.delete_model_version(name, str(version))

    def delete_registered_model(self, name: str) -> None:
        """
        Delete a registered model and all its versions.

        Args:
            name: Model name to delete
        """
        self.client.delete_registered_model(name)

    def set_model_version_tag(
        self, name: str, version: int, key: str, value: str
    ) -> None:
        """
        Set a tag on a model version.

        Args:
            name: Registered model name
            version: Version number
            key: Tag key
            value: Tag value
        """
        self.client.set_model_version_tag(name, str(version), key, value)

    def update_model_version_description(
        self, name: str, version: int, description: str
    ) -> None:
        """
        Update the description of a model version.

        Args:
            name: Registered model name
            version: Version number
            description: New description
        """
        self.client.update_model_version(
            name=name, version=str(version), description=description
        )
