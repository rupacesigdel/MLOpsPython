import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                           precision_score, recall_score, 
                           classification_report)
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "train_data.csv"
MODEL_NAME = "Fraud_Detection_Model"
VALIDATION_THRESHOLD = 0.9
CONFIG = {
    "data": {
        "test_size": 0.2,
        "random_state": 42,
        "target_col": "target"
    },
    "model": {
        "type": "RandomForestClassifier",
        "params": {
            "n_estimators": 150,
            "max_depth": 8,
            "min_samples_split": 2,
            "random_state": 42,
            "class_weight": "balanced"
        }
    }
}

def load_and_validate_data():
    data = pd.read_csv(DATA_PATH)
    
    # Check for any target column if not specified
    target_col = CONFIG['data'].get('target_col', 'target')
    if target_col not in data.columns:
        # Try common target column names
        for col in ['target', 'label', 'class']:
            if col in data.columns:
                target_col = col
                break
        else:
            raise ValueError(f"No target column found in {DATA_PATH}")

    # Adjust sample size check
    min_samples = CONFIG['data'].get('min_samples', 10)
    if len(data) < min_samples:
        logger.warning(f"Dataset has only {len(data)} samples (min {min_samples})")
    
    class_counts = data[CONFIG['data']['target_col']].value_counts()
    if len(class_counts) < 2:
        raise ValueError(f"Need at least 2 classes, found {class_counts.index.tolist()}")
    
    if len(data) < CONFIG['data'].get('min_samples', 100):
        logger.warning(f"Dataset small ({len(data)} samples)")
    
    return train_test_split(
        data.drop(target_col, axis=1),
        data[target_col],
        test_size=CONFIG['data'].get('test_size', 0.2),
        random_state=CONFIG['data'].get('random_state', 42)
    )

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        y_proba = proba[:, -1]
    else:
        y_proba = None
    
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "classification_report": report  # Keep the full report
    }
    
    class_metrics = {
        f"class_{k}_{metric}": v 
        for k, v in report.items() 
        if isinstance(v, dict)
        for metric, v in v.items()
    }
    metrics.update(class_metrics)
    
    if len(np.unique(y_test)) > 1 and y_proba is not None:
        metrics["log_loss"] = log_loss(y_test, y_proba)
    
    return metrics, y_pred

def setup_mlflow():
    """Configure MLflow tracking"""
    mlflow.set_tracking_uri(f"file:{str(BASE_DIR / 'mlruns')}")
    mlflow.set_experiment("Fraud_Detection")
    
    # Enable autologging
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True
    )

def register_and_promote_model(client, run_id, metrics):
    """Handle model versioning and promotion"""
    try:
        # Get the newly created version
        new_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
        
        # Add comprehensive metadata
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=new_version.version,
            key="validation_status",
            value="Pending"
        )
        
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=new_version.version,
            key="deployment_ready",
            value=str(metrics["accuracy"] >= VALIDATION_THRESHOLD).lower()
        )
        
        # Evaluate promotion criteria
        if metrics["accuracy"] >= VALIDATION_THRESHOLD:
            promote_model(client, new_version, metrics)
        else:
            client.set_model_version_tag(
                name=MODEL_NAME,
                version=new_version.version,
                key="validation_status",
                value="Rejected"
            )
            logger.warning(f"Model accuracy {metrics['accuracy']:.2f} below threshold {VALIDATION_THRESHOLD}")
            
    except Exception as e:
        logger.error(f"Model registration failed: {str(e)}")
        raise

def promote_model(client, new_version, metrics):
    """Promote model through staging to production"""
    try:
        # Transition to Staging
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version.version,
            stage="Staging"
        )
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="Challenger",
            version=new_version.version
        )
        
        # Check against current champion
        try:
            champion_version = client.get_model_version_by_alias(MODEL_NAME, "Champion")
            champion_run = client.get_run(champion_version.run_id)
            champion_metrics = champion_run.data.metrics
            
            if metrics["accuracy"] > champion_metrics["accuracy"]:
                # Archive old champion
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=champion_version.version,
                    stage="Archived"
                )
                
                # Promote new champion
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=new_version.version,
                    stage="Production"
                )
                client.set_registered_model_alias(
                    name=MODEL_NAME,
                    alias="Champion",
                    version=new_version.version
                )
                logger.info(f"New champion! Version {new_version.version} promoted to Production")
                
        except Exception as e:
            logger.warning(f"No existing champion found: {str(e)}")
            # First deployment - promote directly to Production
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=new_version.version,
                stage="Production"
            )
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias="Champion",
                version=new_version.version
            )
            
    except Exception as e:
        logger.error(f"Model promotion failed: {str(e)}")
        raise

def train_and_register():
    """End-to-end training and registration pipeline"""
    try:
        # Setup tracking
        setup_mlflow()
        
        # Load data
        X_train, X_test, y_train, y_test = load_and_validate_data()
        
        # Train model
        with mlflow.start_run(run_name=f"challenger_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log config
            mlflow.log_dict(CONFIG, "config.json")
            
            # Initialize model
            model = RandomForestClassifier(**CONFIG['model']['params'])
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics, y_pred = evaluate_model(model, X_test, y_test)
            
            # Log metrics
            mlflow.log_metrics({
                k: v for k, v in metrics.items() 
                if not k.endswith('_report') and isinstance(v, (int, float))
            })

            if 'classification_report' in metrics:
                mlflow.log_text(
                    json.dumps(metrics['classification_report'], indent=2),
                    "classification_report.json"
                )
            
            # Log model
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.iloc[:1],
                registered_model_name=MODEL_NAME
            )
            
            # Register and promote
            client = MlflowClient()
            register_and_promote_model(client, mlflow.active_run().info.run_id, metrics)
            
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    train_and_register()