# XGB tail boom patch v4

Replace the following files in `James1424/XGB_tail_boom_predict_fixed`:

```text
src/model_config.py
src/train_model.py
src/update_model_readme.py
```

## What changed

1. Main model moved to the 3000-round region:

```python
n_estimators = 3000
max_depth = 5
learning_rate = 0.008
```

2. Main feature-weight profile is now:

```python
MAIN_WEIGHT_PROFILE = "core_momentum_aggressive_1"
```

This keeps a single XGB probability model. No hybrid score with raw momentum baseline is added.

3. Hyperparameter ablation now performs a local search around 3000 rounds and lr=0.008:

```text
rounds_2600_lr0008
rounds_2800_lr0008
main_3000_lr0008
rounds_3200_lr0008
rounds_3400_lr0008
rounds_3000_lr0007
rounds_3000_lr0009
rounds_2800_lr0009
rounds_3200_lr0007
```

4. Feature-weight ablation now searches locally around `core_momentum_aggressive_1`, emphasizing `mom_5m`, `mom_4m`, and `core_mom_456_avg` while downweighting non-momentum context:

```text
core_momentum_aggressive_1
local_mom5_456_boost
local_mom4_mom5_boost
local_context_light
local_mom5_dominant
local_mom4_mom5_456_heavy
```

5. Latest live candidates now apply a sanity filter before ranking display:

```text
required: mom_3m, mom_4m, mom_5m, mom_6m, core_mom_456_avg not missing
liquid_vol_score >= 0.50
avg_dollar_volume_3m >= 50,000,000
```

This filter only affects `outputs/latest_live_boom_candidates.csv`; it does not affect training, test metrics, or historical backtests.

6. README formatting was fixed so `max_depth` and `min_child_weight` are not incorrectly rendered as percentages.

## Run

```bash
python run_all.py
```

If the workflow becomes too slow, run one pass with:

```bash
python -m src.train_model --skip-hyperparam-ablation
python -m src.update_model_readme
```
