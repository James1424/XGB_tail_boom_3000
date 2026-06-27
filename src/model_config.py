from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
README_FILE = PROJECT_ROOT / "README.md"

# Full training panel path. In the integrated workflow, build_panel writes this file to outputs/.
# IMPORTANT: panel_head_20000.csv is only a human-readable sample and must never be used for training.
PANEL_FILE = OUTPUT_DIR / "clean_monthly_panel.csv"
PANEL_FALLBACKS = [DATA_DIR / "clean_monthly_panel.csv"]
MIN_TRAINING_PANEL_ROWS = 30000

TARGET_LABEL = "label_boom30_top10_1_3m"
AUX_LABELS = [
    "label_top10_1_3m",
    "label_top5_1_3m",
    "label_boom40_top10_1_3m",
    "label_boom50_top5_1_3m",
    "label_mega100_1_3m",
]
FUTURE_RETURN_COLS = ["future_return_1m", "future_return_2m", "future_return_3m", "future_max_return_1_3m"]
LEAKAGE_COLUMNS = FUTURE_RETURN_COLS + [
    "future_max_return_1_3m_pct_rank",
    "monthly_top10_threshold_1_3m",
    "monthly_top5_threshold_1_3m",
    TARGET_LABEL,
] + AUX_LABELS
ID_COLUMNS = ["month", "ticker", "adj_close", "sources", "categories"]

TRAIN_END = "2021-12-31"
VALID_START = "2022-01-01"
VALID_END = "2023-12-31"
TEST_START = "2024-01-01"

TOP_KS = [3, 5, 10]
MAIN_SEEDS = [7, 42, 202, 777, 2026]
TRAINING_CURVE_ROUNDS = list(range(100, 3001, 100))

# Main model configuration: switch production baseline to the 3000-round region.
MODEL_PARAMS = {
    "n_estimators": 3000,
    "max_depth": 5,
    "learning_rate": 0.008,
    "subsample": 0.85,
    "colsample_bytree": 0.90,
    "colsample_bylevel": 0.85,
    "colsample_bynode": 0.85,
    "min_child_weight": 3,
    "reg_alpha": 0.05,
    "reg_lambda": 2.00,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

# Local hyperparameter ablation around 3000 rounds and lr=0.008.
HYPERPARAMETER_ABLATION_PROFILES = {
    "rounds_2600_lr0008": {**MODEL_PARAMS, "n_estimators": 2600, "learning_rate": 0.008},
    "rounds_2800_lr0008": {**MODEL_PARAMS, "n_estimators": 2800, "learning_rate": 0.008},
    "main_3000_lr0008": dict(MODEL_PARAMS),
    "rounds_3200_lr0008": {**MODEL_PARAMS, "n_estimators": 3200, "learning_rate": 0.008},
    "rounds_3400_lr0008": {**MODEL_PARAMS, "n_estimators": 3400, "learning_rate": 0.008},
    "rounds_3000_lr0007": {**MODEL_PARAMS, "n_estimators": 3000, "learning_rate": 0.007},
    "rounds_3000_lr0009": {**MODEL_PARAMS, "n_estimators": 3000, "learning_rate": 0.009},
    "rounds_2800_lr0009": {**MODEL_PARAMS, "n_estimators": 2800, "learning_rate": 0.009},
    "rounds_3200_lr0007": {**MODEL_PARAMS, "n_estimators": 3200, "learning_rate": 0.007},
}

ABLATION_GROUPS = {
    "core_momentum": ["mom_4m", "mom_5m", "mom_6m", "core_mom_456", "mom_6m_"],
    "other_momentum": ["mom_1m", "mom_2m", "mom_3m", "mom_7m", "mom_9m", "mom_12m"],
    "relative_strength": ["rel_mom_"],
    "trend": ["price_ma", "ma5_slope", "ma10_slope", "ma20_slope", "ma30_slope", "ma50_slope", "ma100_slope"],
    "risk_drawdown": ["drawdown", "volatility_", "return_vol_ratio"],
    "volatility_frequency": ["large_move_freq", "up_big_move_freq", "down_big_move_freq", "avg_abs_daily_return", "intraday_range"],
    "liquidity_size": ["avg_dollar_volume", "dollar_volume", "trading_day_count", "liquid_vol_score"],
    "volume_flow": ["volume_change", "volume_ratio", "volume_ma", "up_day_volume", "up_day_dollar"],
    "qqq_context": ["qqq_mom_"],
    "etf_source": ["source_count", "source_weight_sum", "theme_count", "in_"],
}

# Feature-weight profiles: no hybrid ranking, but much stronger prior on mom_4m/mom_5m/core_mom_456_avg.
# The main profile is the previously best local row, then four/five profiles search nearby.
FEATURE_WEIGHT_PROFILES = {
    "core_momentum_aggressive_1": {
        "core_momentum": 7.00,
        "relative_strength": 1.00,
        "volatility_frequency": 0.90,
        "liquidity_size": 0.85,
        "other_momentum": 0.80,
        "trend": 0.70,
        "risk_drawdown": 0.65,
        "volume_flow": 0.65,
        "qqq_context": 0.60,
        "etf_source": 0.55,
        "unclassified": 0.85,
        "_feature_overrides": {
            "mom_4m": 16.00,
            "mom_5m": 24.00,
            "mom_6m": 24.00,
            "core_mom_456_avg": 32.00,
            "core_mom_456_min": 14.00,
            "core_mom_456_max": 14.00,
            "core_mom_456_std": 5.00,
            "mom_4m_vs_6m": 7.00,
            "mom_5m_vs_6m": 7.00,
            "mom_6m_first3m": 10.00,
            "mom_6m_last3m": 10.00,
            "mom_6m_acceleration": 12.00,
        },
    },
    "local_mom5_456_boost": {
        "core_momentum": 8.00,
        "relative_strength": 0.95,
        "volatility_frequency": 0.85,
        "liquidity_size": 0.80,
        "other_momentum": 0.75,
        "trend": 0.65,
        "risk_drawdown": 0.60,
        "volume_flow": 0.60,
        "qqq_context": 0.55,
        "etf_source": 0.50,
        "unclassified": 0.80,
        "_feature_overrides": {
            "mom_4m": 17.00,
            "mom_5m": 30.00,
            "mom_6m": 24.00,
            "core_mom_456_avg": 40.00,
            "core_mom_456_min": 15.00,
            "core_mom_456_max": 15.00,
            "core_mom_456_std": 5.00,
            "mom_4m_vs_6m": 7.00,
            "mom_5m_vs_6m": 8.00,
            "mom_6m_first3m": 10.00,
            "mom_6m_last3m": 10.00,
            "mom_6m_acceleration": 12.00,
        },
    },
    "local_mom4_mom5_boost": {
        "core_momentum": 8.00,
        "relative_strength": 0.95,
        "volatility_frequency": 0.85,
        "liquidity_size": 0.80,
        "other_momentum": 0.75,
        "trend": 0.65,
        "risk_drawdown": 0.60,
        "volume_flow": 0.60,
        "qqq_context": 0.55,
        "etf_source": 0.50,
        "unclassified": 0.80,
        "_feature_overrides": {
            "mom_4m": 24.00,
            "mom_5m": 32.00,
            "mom_6m": 22.00,
            "core_mom_456_avg": 36.00,
            "core_mom_456_min": 15.00,
            "core_mom_456_max": 15.00,
            "core_mom_456_std": 5.00,
            "mom_4m_vs_6m": 8.00,
            "mom_5m_vs_6m": 8.00,
            "mom_6m_first3m": 9.00,
            "mom_6m_last3m": 9.00,
            "mom_6m_acceleration": 11.00,
        },
    },
    "local_context_light": {
        "core_momentum": 9.00,
        "relative_strength": 0.80,
        "volatility_frequency": 0.70,
        "liquidity_size": 0.70,
        "other_momentum": 0.65,
        "trend": 0.55,
        "risk_drawdown": 0.50,
        "volume_flow": 0.50,
        "qqq_context": 0.45,
        "etf_source": 0.40,
        "unclassified": 0.70,
        "_feature_overrides": {
            "mom_4m": 20.00,
            "mom_5m": 32.00,
            "mom_6m": 28.00,
            "core_mom_456_avg": 42.00,
            "core_mom_456_min": 16.00,
            "core_mom_456_max": 16.00,
            "core_mom_456_std": 5.00,
            "mom_4m_vs_6m": 8.00,
            "mom_5m_vs_6m": 8.00,
            "mom_6m_first3m": 11.00,
            "mom_6m_last3m": 11.00,
            "mom_6m_acceleration": 13.00,
        },
    },
    "local_mom5_dominant": {
        "core_momentum": 9.00,
        "relative_strength": 0.75,
        "volatility_frequency": 0.65,
        "liquidity_size": 0.65,
        "other_momentum": 0.60,
        "trend": 0.50,
        "risk_drawdown": 0.45,
        "volume_flow": 0.45,
        "qqq_context": 0.40,
        "etf_source": 0.35,
        "unclassified": 0.65,
        "_feature_overrides": {
            "mom_4m": 22.00,
            "mom_5m": 42.00,
            "mom_6m": 24.00,
            "core_mom_456_avg": 44.00,
            "core_mom_456_min": 16.00,
            "core_mom_456_max": 16.00,
            "core_mom_456_std": 5.00,
            "mom_4m_vs_6m": 8.00,
            "mom_5m_vs_6m": 10.00,
            "mom_6m_first3m": 10.00,
            "mom_6m_last3m": 10.00,
            "mom_6m_acceleration": 12.00,
        },
    },
    "local_mom4_mom5_456_heavy": {
        "core_momentum": 10.00,
        "relative_strength": 0.70,
        "volatility_frequency": 0.60,
        "liquidity_size": 0.60,
        "other_momentum": 0.55,
        "trend": 0.45,
        "risk_drawdown": 0.40,
        "volume_flow": 0.40,
        "qqq_context": 0.35,
        "etf_source": 0.30,
        "unclassified": 0.60,
        "_feature_overrides": {
            "mom_4m": 30.00,
            "mom_5m": 44.00,
            "mom_6m": 28.00,
            "core_mom_456_avg": 55.00,
            "core_mom_456_min": 18.00,
            "core_mom_456_max": 18.00,
            "core_mom_456_std": 5.00,
            "mom_4m_vs_6m": 10.00,
            "mom_5m_vs_6m": 10.00,
            "mom_6m_first3m": 12.00,
            "mom_6m_last3m": 12.00,
            "mom_6m_acceleration": 14.00,
        },
    },
}
MAIN_WEIGHT_PROFILE = "core_momentum_aggressive_1"
FEATURE_GROUP_WEIGHTS = FEATURE_WEIGHT_PROFILES[MAIN_WEIGHT_PROFILE]

# Live candidate filters. These only affect outputs/latest_live_boom_candidates.csv.
LIVE_REQUIRED_FEATURES = ["mom_3m", "mom_4m", "mom_5m", "mom_6m", "core_mom_456_avg"]
LIVE_MIN_LIQUID_VOL_SCORE = 0.50
LIVE_MIN_AVG_DOLLAR_VOLUME_3M = 50_000_000

OUTPUT_FILES = {
    "final_metrics": OUTPUT_DIR / "final_train_validation_test_metrics.csv",
    "main_result": OUTPUT_DIR / "reference_downweighted_main_model_result.csv",
    "strategy_baseline": OUTPUT_DIR / "strategy_baseline_comparison.csv",
    "latest_live": OUTPUT_DIR / "latest_live_boom_candidates.csv",
    "recent_top3": OUTPUT_DIR / "recent_xgb_top3_backtest_months.csv",
    "ablation": OUTPUT_DIR / "ablation_ranked_summary.csv",
    "five_seed": OUTPUT_DIR / "five_seed_training_stability.csv",
    "training_curve": OUTPUT_DIR / "training_curve_metrics_every_100_rounds.csv",
    "five_seed_feature_importance": OUTPUT_DIR / "five_seed_average_feature_importance.csv",
    "manual_feature_weights": OUTPUT_DIR / "manual_feature_weights_used_by_xgboost.csv",
    "feature_weight_ablation": OUTPUT_DIR / "feature_weight_ablation_summary.csv",
    "hyperparameter_ablation": OUTPUT_DIR / "hyperparameter_ablation_summary.csv",
    "monthly_top": OUTPUT_DIR / "monthly_top_predictions.csv",
    "full_predictions": OUTPUT_DIR / "full_predictions.csv",
    "metrics_json": OUTPUT_DIR / "model_metrics.json",
}
MAIN_MODEL_FILE = MODEL_DIR / "xgb_tail_event_classifier.json"
FEATURE_LIST_FILE = MODEL_DIR / "selected_features.txt"
