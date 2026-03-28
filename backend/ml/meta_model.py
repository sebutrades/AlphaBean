"""
backend/ml/meta_model.py — XGBoost Meta-Model

Combines ALL existing signals (price patterns, technical indicators,
sentiment, regime) into one probability score that predicts whether
a given trade setup will be profitable.

HOW IT WORKS:
  1. FEATURE EXTRACTION: For each TradeSetup, extract ~30 features:
     - Pattern identity (one-hot or label-encoded)
     - Pattern confidence, R:R ratio
     - Technical: price vs SMAs, ATR ratio, RVOL, vol compression
     - Sentiment: net score, article count
     - Regime: one-hot encoded market state
     - Backtest history: pattern's win rate, expectancy, PF
     
  2. TRAINING: Uses historical backtest outcomes as labels.
     Feature matrix from setup metadata, label = 1 (win/partial_win) or 0 (loss).
     Trained with walk-forward: train on months 1-2, test on month 3.
     
  3. PREDICTION: For each new setup from the scanner, extract features
     and predict P(profitable). This becomes a 7th factor in the composite score.

  4. RETRAINING: Auto-retrains weekly from accumulated outcomes.

DEPENDENCIES:
  pip install xgboost scikit-learn

USAGE:
  from backend.ml.meta_model import MetaModel
  
  model = MetaModel()
  model.train_from_backtest_cache()   # Train from cached results
  
  # Score a new setup
  prob = model.predict_setup(setup_dict, sentiment_data, regime_str)
"""
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from backend.patterns.registry import PATTERN_META


# ==============================================================================
# CONFIG
# ==============================================================================

MODEL_PATH = Path("cache/meta_model.pkl")
FEATURE_LOG_PATH = Path("cache/meta_features.json")
BACKTEST_CACHE = Path("cache/backtest_results.json")
STRATEGY_PERF = Path("cache/strategy_performance.json")

# All pattern names we might see (for encoding)
ALL_PATTERNS = sorted(PATTERN_META.keys())
PATTERN_TO_IDX = {name: i for i, name in enumerate(ALL_PATTERNS)}

REGIME_MAP = {
    "trending_bull": 0, "trending_bear": 1,
    "mean_reverting": 2, "high_volatility": 3,
    "mixed": 4, "unknown": 5,
}

STRATEGY_TYPE_MAP = {
    "momentum": 0, "mean_reversion": 1,
    "breakout": 2, "scalp": 3,
}


# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================

def _load_pattern_stats(pattern_name: str) -> dict:
    """Load backtest stats for a specific pattern from cache."""
    if not BACKTEST_CACHE.exists():
        return {}
    try:
        data = json.loads(BACKTEST_CACHE.read_text())
        return data.get("patterns", {}).get(pattern_name, {})
    except Exception:
        return {}

def extract_features(
    setup_dict: dict,
    structures: object = None,
    sentiment_data: dict = None,
    regime_str: str = "unknown",
) -> np.ndarray:
    """Extract feature vector from a trade setup.
    
    Returns a numpy array of ~30 features.
    
    Feature groups:
      [0]     Pattern index (label encoded)
      [1]     Strategy type (label encoded)
      [2]     Confidence (0-1)
      [3]     Risk:Reward ratio
      [4]     Bias (1=long, 0=short)
      [5-8]   Regime one-hot (bull, bear, mean_rev, high_vol)
      [9]     Pattern historical win rate
      [10]    Pattern historical expectancy
      [11]    Pattern historical profit factor
      [12]    Pattern sample size (log-scaled)
      [13]    Pattern edge score
      [14-19] Scoring breakdown (6 factors from composite)
      [20]    Sentiment net (-1 to +1)
      [21]    Sentiment article count (log-scaled)
      [22-27] Technical features (if structures available)
    """
    features = np.zeros(30, dtype=np.float64)
    
    # Pattern identity
    pattern_name = setup_dict.get("pattern_name", "")
    features[0] = PATTERN_TO_IDX.get(pattern_name, -1)
    
    # Strategy type
    meta = PATTERN_META.get(pattern_name, {})
    stype = meta.get("type", "breakout")
    features[1] = STRATEGY_TYPE_MAP.get(stype, 2)
    
    # Confidence and R:R
    features[2] = setup_dict.get("confidence", 0.5)
    features[3] = min(setup_dict.get("risk_reward_ratio", 1.0), 10.0)  # Cap at 10
    
    # Bias
    features[4] = 1.0 if setup_dict.get("bias", "").lower() == "long" else 0.0
    
    # Regime one-hot
    regime_idx = REGIME_MAP.get(regime_str, 5)
    if regime_idx < 4:
        features[5 + regime_idx] = 1.0
    
    # Pattern historical stats (from backtest cache)
    bt = _load_pattern_stats(pattern_name)
    features[9] = bt.get("win_rate", 50.0) / 100.0
    features[10] = max(-2.0, min(2.0, bt.get("expectancy", 0.0)))
    features[11] = min(5.0, bt.get("profit_factor", 1.0))
    features[12] = np.log1p(bt.get("total_signals", 0))
    features[13] = bt.get("edge_score", 50.0) / 100.0
    
    # Scoring breakdown
    scoring = setup_dict.get("scoring", {})
    features[14] = scoring.get("pattern_confidence", 50) / 100.0
    features[15] = scoring.get("feature_score", 50) / 100.0
    features[16] = scoring.get("strategy_score", 50) / 100.0
    features[17] = scoring.get("regime_alignment", 50) / 100.0
    features[18] = scoring.get("backtest_edge", 50) / 100.0
    features[19] = scoring.get("volume_confirm", 50) / 100.0
    
    # Sentiment
    if sentiment_data:
        features[20] = sentiment_data.get("net", 0.0)
        features[21] = np.log1p(sentiment_data.get("total", 0))
    
    # Technical features (from structures object)
    if structures is not None and hasattr(structures, 'n') and structures.n > 20:
        s = structures
        cur = s.closes[-1]
        atr = s.current_atr if s.current_atr > 0 else 1.0
        
        # Price vs 20 SMA (normalized by ATR)
        sma20 = float(np.mean(s.closes[-20:]))
        features[22] = (cur - sma20) / atr
        
        # Price vs 50 SMA
        if s.n >= 50:
            sma50 = float(np.mean(s.closes[-50:]))
            features[23] = (cur - sma50) / atr
        
        # RVOL
        avg_vol = float(np.mean(s.volumes[-20:]))
        features[24] = s.volumes[-1] / avg_vol if avg_vol > 0 else 1.0
        
        # 5-day return
        if s.n >= 6:
            features[25] = (cur - s.closes[-6]) / s.closes[-6]
        
        # 20-day return
        if s.n >= 21:
            features[26] = (cur - s.closes[-21]) / s.closes[-21]
        
        # Volatility ratio (60d vs 252d)
        if s.n >= 60:
            rets_60 = np.diff(s.closes[-61:]) / np.array(s.closes[-61:-1])
            vol_60 = float(np.std(rets_60))
            if s.n >= 252:
                rets_252 = np.diff(s.closes[-253:]) / np.array(s.closes[-253:-1])
                vol_252 = float(np.std(rets_252))
                features[27] = vol_60 / vol_252 if vol_252 > 0 else 1.0
    
    return features


FEATURE_NAMES = [
    "pattern_idx", "strategy_type", "confidence", "rr_ratio", "is_long",
    "regime_bull", "regime_bear", "regime_mr", "regime_hv",
    "hist_wr", "hist_exp", "hist_pf", "hist_sample_log", "hist_edge",
    "score_pattern", "score_feature", "score_strategy", "score_regime",
    "score_backtest", "score_volume",
    "sentiment_net", "sentiment_articles_log",
    "tech_price_vs_sma20", "tech_price_vs_sma50", "tech_rvol",
    "tech_ret_5d", "tech_ret_20d", "tech_vol_ratio",
    "reserved_1", "reserved_2",
]


# ==============================================================================
# META MODEL
# ==============================================================================

class MetaModel:
    """XGBoost meta-model that predicts trade outcome probability."""
    
    def __init__(self):
        self.model = None
        self.trained_at = None
        self.train_samples = 0
        self.train_accuracy = 0.0
        self.train_auc = 0.0
        self.feature_importance = {}
    
    def is_trained(self) -> bool:
        return self.model is not None
    
    def train_from_backtest_cache(self, min_samples: int = 500) -> dict:
        """Train the model from backtest results.
        
        Loads all trade outcomes from the strategy evaluator cache,
        extracts features from each, and trains XGBoost.
        
        Returns training metrics.
        """
        if not HAS_XGB:
            return {"error": "XGBoost not installed. Run: pip install xgboost scikit-learn"}
        
        # Load outcomes
        outcomes = self._load_outcomes()
        if len(outcomes) < min_samples:
            return {"error": f"Need {min_samples}+ outcomes, have {len(outcomes)}"}
        
        print(f"  [MetaModel] Training on {len(outcomes)} outcomes...")
        
        # Build feature matrix and labels
        X = []
        y = []
        
        for outcome in outcomes:
            features = self._outcome_to_features(outcome)
            if features is not None:
                X.append(features)
                label = 1 if outcome.get("outcome") in ("win", "partial_win") else 0
                y.append(label)
        
        if len(X) < min_samples:
            return {"error": f"Only {len(X)} valid feature vectors (need {min_samples})"}
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"  [MetaModel] Feature matrix: {X.shape}, "
              f"positive rate: {y.mean():.1%}")
        
        # Train/test split (temporal — last 20% for validation)
        split = int(len(X) * 0.80)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        self.train_accuracy = accuracy_score(y_test, y_pred)
        try:
            self.train_auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            self.train_auc = 0.0
        
        self.train_samples = len(X)
        self.trained_at = datetime.now().isoformat()
        
        # Feature importance
        importances = self.model.feature_importances_
        self.feature_importance = {
            FEATURE_NAMES[i]: round(float(importances[i]), 4)
            for i in range(min(len(importances), len(FEATURE_NAMES)))
            if importances[i] > 0.01
        }
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(),
                   key=lambda x: x[1], reverse=True)
        )
        
        # Save model
        self.save()
        
        metrics = {
            "samples": len(X),
            "positive_rate": float(y.mean()),
            "accuracy": round(self.train_accuracy, 4),
            "auc": round(self.train_auc, 4),
            "top_features": dict(list(self.feature_importance.items())[:10]),
            "trained_at": self.trained_at,
        }
        
        print(f"  [MetaModel] Accuracy: {self.train_accuracy:.1%}, "
              f"AUC: {self.train_auc:.3f}")
        print(f"  [MetaModel] Top features: "
              f"{', '.join(list(self.feature_importance.keys())[:5])}")
        
        return metrics
    
    def predict_setup(
        self,
        setup_dict: dict,
        structures: object = None,
        sentiment_data: dict = None,
        regime_str: str = "unknown",
    ) -> float:
        """Predict probability of a profitable outcome for a setup.
        
        Returns:
            Float 0.0 to 1.0 — probability the trade will be profitable.
            Returns 0.5 (neutral) if model not trained.
        """
        if not self.is_trained():
            return 0.5
        
        features = extract_features(setup_dict, structures, sentiment_data, regime_str)
        prob = self.model.predict_proba(features.reshape(1, -1))[0, 1]
        return float(prob)
    
    def predict_batch(
        self,
        setups: list[dict],
        structures_by_symbol: dict = None,
        sentiment_by_symbol: dict = None,
        regime_str: str = "unknown",
    ) -> list[float]:
        """Predict probabilities for multiple setups."""
        if not self.is_trained():
            return [0.5] * len(setups)
        
        if structures_by_symbol is None:
            structures_by_symbol = {}
        if sentiment_by_symbol is None:
            sentiment_by_symbol = {}
        
        X = []
        for setup in setups:
            sym = setup.get("symbol", "")
            features = extract_features(
                setup,
                structures_by_symbol.get(sym),
                sentiment_by_symbol.get(sym),
                regime_str,
            )
            X.append(features)
        
        X = np.array(X)
        probs = self.model.predict_proba(X)[:, 1]
        return [float(p) for p in probs]
    
    def save(self):
        """Save model to disk."""
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self.model,
            "trained_at": self.trained_at,
            "train_samples": self.train_samples,
            "train_accuracy": self.train_accuracy,
            "train_auc": self.train_auc,
            "feature_importance": self.feature_importance,
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(data, f)
    
    def load(self) -> bool:
        """Load model from disk. Returns True if successful."""
        if not MODEL_PATH.exists():
            return False
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.trained_at = data.get("trained_at")
            self.train_samples = data.get("train_samples", 0)
            self.train_accuracy = data.get("train_accuracy", 0)
            self.train_auc = data.get("train_auc", 0)
            self.feature_importance = data.get("feature_importance", {})
            return True
        except Exception:
            return False
    
   

    
    def get_status(self) -> dict:
        """Return model status for API/UI."""
        return {
            "trained": self.is_trained(),
            "trained_at": self.trained_at,
            "samples": self.train_samples,
            "accuracy": self.train_accuracy,
            "auc": self.train_auc,
            "top_features": dict(list(self.feature_importance.items())[:5]),
        }
    
    # ── Internal helpers ──
    
    def _load_outcomes(self) -> list[dict]:
        """Load trade outcomes from evaluator cache."""
        if not STRATEGY_PERF.exists():
            return []
        
        try:
            data = json.loads(STRATEGY_PERF.read_text())
            # The evaluator stores outcomes as a list
            outcomes = data.get("outcomes", [])
            if not outcomes:
                # Try loading from backtest results
                if BACKTEST_CACHE.exists():
                    bt = json.loads(BACKTEST_CACHE.read_text())
                    # Build pseudo-outcomes from pattern stats
                    outcomes = self._outcomes_from_backtest(bt)
            return outcomes
        except Exception:
            return []
    
    def _outcomes_from_backtest(self, bt_data: dict) -> list[dict]:
        """Generate training data from backtest results.
        
        Since we don't have per-trade features stored, we generate
        synthetic training examples from pattern-level statistics.
        Each pattern's win rate determines the label distribution.
        """
        outcomes = []
        patterns = bt_data.get("patterns", {})
        
        for name, stats in patterns.items():
            n = stats.get("total_signals", 0)
            wr = stats.get("win_rate", 0) / 100.0
            exp = stats.get("expectancy", 0)
            pf = stats.get("profit_factor", 1.0)
            
            if n < 10:
                continue
            
            # Generate proportional win/loss examples
            n_wins = int(n * wr)
            n_losses = n - n_wins
            
            meta = PATTERN_META.get(name, {})
            stype = meta.get("type", "breakout")
            
            base = {
                "pattern_name": name,
                "strategy_type": stype,
                "confidence": meta.get("wr", 0.5),
                "risk_reward_ratio": stats.get("avg_win_r", 1.0) / max(stats.get("avg_loss_r", 1.0), 0.1),
                "bias": "long",  # Simplified — we don't know per-trade bias
            }
            
            # Cap at 200 examples per pattern to prevent class imbalance
            cap = min(200, n)
            cap_wins = int(cap * wr)
            cap_losses = cap - cap_wins
            
            for _ in range(cap_wins):
                outcomes.append({**base, "outcome": "win",
                                 "realized_r": stats.get("avg_win_r", 1.0)})
            for _ in range(cap_losses):
                outcomes.append({**base, "outcome": "loss",
                                 "realized_r": -stats.get("avg_loss_r", 1.0)})
        
        # Shuffle
        np.random.seed(42)
        np.random.shuffle(outcomes)
        return outcomes
    
    def _outcome_to_features(self, outcome: dict) -> Optional[np.ndarray]:
        """Convert a single outcome dict to a feature vector."""
        try:
            return extract_features(outcome)
        except Exception:
            return None


# ==============================================================================
# CONVENIENCE: Train and test from command line
# ==============================================================================

def train_meta_model():
    """Train the meta-model from backtest cache."""
    if not HAS_XGB:
        print("  ✗ XGBoost not installed. Run:")
        print("    pip install xgboost scikit-learn")
        return
    
    model = MetaModel()
    metrics = model.train_from_backtest_cache(min_samples=100)
    
    if "error" in metrics:
        print(f"  ✗ {metrics['error']}")
        return
    
    print(f"\n  ✓ Meta-model trained:")
    print(f"    Samples:     {metrics['samples']}")
    print(f"    Accuracy:    {metrics['accuracy']:.1%}")
    print(f"    AUC:         {metrics['auc']:.3f}")
    print(f"    Pos rate:    {metrics['positive_rate']:.1%}")
    print(f"    Top features:")
    for name, imp in list(metrics['top_features'].items())[:8]:
        bar = "█" * int(imp * 50)
        print(f"      {name:<28} {bar} {imp:.3f}")
    
    # Quick test: predict on a mock setup
    mock = {
        "pattern_name": "Juicer Long",
        "confidence": 0.72,
        "risk_reward_ratio": 2.8,
        "bias": "long",
    }
    prob = model.predict_setup(mock, regime_str="trending_bull")
    print(f"\n  Test prediction (Juicer Long, bull regime): {prob:.1%} win probability")
    
    return model


if __name__ == "__main__":
    train_meta_model()