# alias-based "promotion" (no stages)
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "event_classifier"
ALIAS = "staging"

def main():
    client = MlflowClient()

    # Get all versions and pick the highest version number
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise SystemExit(f"No versions found for '{MODEL_NAME}'. Run training first.")

    latest = max(int(v.version) for v in versions)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, latest)
    print(f"✅ Set alias '{ALIAS}' → {MODEL_NAME} v{latest}")

if __name__ == "__main__":
    main()
