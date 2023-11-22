# intro
based on Madewithml.com structure and content. It is adapted for our purpose in the DAIS project.

First run this on command line. To setup a new virtual env and all the needed python libs.
```
export PYTHONPATH=$PYTHONPATH:$PWD
python3 -m venv venv  # recommend using Python 3.10
source venv/bin/activate  # on Windows: venv\Scripts\activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
pre-commit install
pre-commit autoupdate
```

Once everything installed. Let's start to work (considering the code is run locally, look at the original github for anyscale support):

# Training
```
export EXPERIMENT_NAME="fatigue_with_CV"
export DATASET_LOC="/Volumes/Elements/dais/data/uta_rldd/processed/"
export TRAIN_LOOP_CONFIG='{"fc_size": 512, "lr": 1e-3, "lr_factor": 0.8, "lr_patience": 3}'
python train.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-loc "$DATASET_LOC" \
    --train-loop-config "$TRAIN_LOOP_CONFIG" \
    --num-samples 1000 \
    --num-workers 4 \
    --cpu-per-worker 1 \
    --gpu-per-worker 0 \
    --num-epochs 15 \
    --batch-size 16 \
    --results-fp results/training_results.json
```

# Tuning 
```
export EXPERIMENT_NAME="fatigue_with_CV"
export DATASET_LOC="/Volumes/Elements/dais/data/uta_rldd/processed/"
export TRAIN_LOOP_CONFIG='{"fc_size": 512, "lr": 1e-3, "lr_factor": 0.8, "lr_patience": 3}'
export INITIAL_PARAMS="[{\"train_loop_config\": $TRAIN_LOOP_CONFIG}]"
python tune.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --dataset-loc "$DATASET_LOC" \
    --initial-params "$INITIAL_PARAMS" \
    --num-runs 50 \
    --num-workers 6 \
    --cpu-per-worker 1 \
    --gpu-per-worker 0 \
    --num-epochs 15 \
    --batch-size 16 \
    --results-fp results/tuning_results.json
```

# Experiment Tracking
Use the MLflow library to track our experiments and store our models and the MLflow Tracking UI to view our experiments. We have been saving our experiments to a local directory but note that in an actual production setting, we would have a central location to store all of our experiments. It's easy/inexpensive to spin up your own MLflow server for all of your team members to track their experiments on or use a managed solution like Weights & Biases, Comet, etc.

```
export MODEL_REGISTRY=$(python -c "import config; print(config.MODEL_REGISTRY)")
mlflow server -h 0.0.0.0 -p 8080 --backend-store-uri $MODEL_REGISTRY
```
Then look at http://localhost:8080/ to view your MLflow dashboard.

# Evaluation
```
export EXPERIMENT_NAME="fatigue_with_CV"
export RUN_ID=$(python predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
export HOLDOUT_LOC="/Volumes/Elements/dais/data/uta_rldd/processed/"
python evaluate.py \
    --run-id $RUN_ID \
    --dataset-loc "$HOLDOUT_LOC" \
    --participant 0 \
    --subset "test" \
    --results-fp results/evaluation_results.json
```

# Inference
```
# Get run ID
export EXPERIMENT_NAME="fatigue_with_CV"
export DATASET_LOC="/Volumes/Elements/dais/data/uta_rldd/processed/"
export RUN_ID=$(python predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
python predict.py predict \
    --run-id $RUN_ID \
    --dataset-loc "$DATASET_LOC" \
    --participant 0 \
    --subset "test" 
```

# Serve

## Start the Ray server
```
ray start --head
```

Then launch the serve script:
```
# Set up
export EXPERIMENT_NAME="fatigue_with_CV"
export RUN_ID=$(python predict.py get-best-run-id --experiment-name $EXPERIMENT_NAME --metric val_loss --mode ASC)
python serve.py --run_id $RUN_ID
```

On the client side, use python or curl. Here an example in python to request the service:
```
import numpy as np
import requests
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
       
feats = np.random.randn(16000)
json_data = json.dumps({"features": feats}, cls=NpEncoder)
requests.post("http://127.0.0.1:8000/predict", data=json_data).json()
# works as well with evaluate as an endpoint but needs a dataset
```

then stop all the server processes
```
ray stop
```
