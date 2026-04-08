"""
Lightweight transformer for next-day SPY direction prediction.

Pure numpy implementation (no PyTorch dependency).
4 layers, 64-dim, 4 heads, 20-day lookback, causal masking.

Training: evolutionary parameter search (gradient-free) on walk-forward
windows. Compared against XGBoost baseline.

Usage::

    from compass.transformer_predictor import TransformerPredictor
    tp = TransformerPredictor()
    result = tp.train_and_evaluate(features_df, targets)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "transformer_predictor.html"
TRADING_DAYS = 252


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class TransformerConfig:
    seq_len: int = 20       # lookback window
    d_model: int = 64       # hidden dimension
    n_heads: int = 4        # attention heads
    n_layers: int = 4       # encoder layers
    d_ff: int = 128         # feedforward hidden dim
    dropout: float = 0.0    # no dropout in numpy impl
    n_features: int = 8     # input features per timestep


# ── Numpy transformer components ─────────────────────────────────────────


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding."""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[:d_model // 2])
    return pe


def causal_mask(seq_len: int) -> np.ndarray:
    """Lower-triangular causal mask. 1 = attend, 0 = mask."""
    return np.tril(np.ones((seq_len, seq_len)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


# ── Transformer layer (single sample) ───────────────────────────────────


@dataclass
class TransformerWeights:
    """All weights for the transformer."""

    # Input projection
    W_in: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    # Per-layer weights
    W_q: List[np.ndarray] = field(default_factory=list)  # [n_layers] each (d_model, d_model)
    W_k: List[np.ndarray] = field(default_factory=list)
    W_v: List[np.ndarray] = field(default_factory=list)
    W_o: List[np.ndarray] = field(default_factory=list)
    W_ff1: List[np.ndarray] = field(default_factory=list)  # (d_model, d_ff)
    W_ff2: List[np.ndarray] = field(default_factory=list)  # (d_ff, d_model)
    # Output head
    W_out: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    b_out: float = 0.0


def init_weights(cfg: TransformerConfig, seed: int = 42) -> TransformerWeights:
    """Xavier-style initialization."""
    rng = np.random.RandomState(seed)
    scale = lambda fan_in, fan_out: math.sqrt(2.0 / (fan_in + fan_out))

    w = TransformerWeights()
    s = scale(cfg.n_features, cfg.d_model)
    w.W_in = rng.randn(cfg.n_features, cfg.d_model) * s

    for _ in range(cfg.n_layers):
        s = scale(cfg.d_model, cfg.d_model)
        w.W_q.append(rng.randn(cfg.d_model, cfg.d_model) * s)
        w.W_k.append(rng.randn(cfg.d_model, cfg.d_model) * s)
        w.W_v.append(rng.randn(cfg.d_model, cfg.d_model) * s)
        w.W_o.append(rng.randn(cfg.d_model, cfg.d_model) * s)
        s_ff = scale(cfg.d_model, cfg.d_ff)
        w.W_ff1.append(rng.randn(cfg.d_model, cfg.d_ff) * s_ff)
        s_ff2 = scale(cfg.d_ff, cfg.d_model)
        w.W_ff2.append(rng.randn(cfg.d_ff, cfg.d_model) * s_ff2)

    w.W_out = rng.randn(cfg.d_model, 1) * scale(cfg.d_model, 1)
    w.b_out = 0.0
    return w


def multi_head_attention(
    x: np.ndarray,  # (seq_len, d_model)
    W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray, W_o: np.ndarray,
    n_heads: int,
    mask: np.ndarray,
) -> np.ndarray:
    """Multi-head self-attention with causal mask."""
    seq_len, d_model = x.shape
    d_head = d_model // n_heads

    Q = x @ W_q  # (seq, d_model)
    K = x @ W_k
    V = x @ W_v

    # Reshape to (n_heads, seq, d_head)
    Q = Q.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

    # Attention scores
    scores = Q @ K.transpose(0, 2, 1) / math.sqrt(d_head)  # (heads, seq, seq)
    # Apply causal mask
    scores = scores * mask[np.newaxis, :, :] + (1 - mask[np.newaxis, :, :]) * (-1e9)
    attn = softmax(scores, axis=-1)

    # Weighted values
    out = attn @ V  # (heads, seq, d_head)
    out = out.transpose(1, 0, 2).reshape(seq_len, d_model)
    return out @ W_o


def feedforward(x: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    return gelu(x @ W1) @ W2


def transformer_forward(
    x: np.ndarray,  # (seq_len, n_features)
    weights: TransformerWeights,
    cfg: TransformerConfig,
) -> float:
    """Forward pass returning P(up) for the last position."""
    seq_len = x.shape[0]

    # Input projection + positional encoding
    h = x @ weights.W_in  # (seq, d_model)
    h = h + positional_encoding(seq_len, cfg.d_model)[:seq_len]

    mask = causal_mask(seq_len)

    # Transformer layers
    for i in range(cfg.n_layers):
        # Self-attention + residual + norm
        attn = multi_head_attention(
            h, weights.W_q[i], weights.W_k[i], weights.W_v[i], weights.W_o[i],
            cfg.n_heads, mask,
        )
        h = layer_norm(h + attn)

        # Feedforward + residual + norm
        ff = feedforward(h, weights.W_ff1[i], weights.W_ff2[i])
        h = layer_norm(h + ff)

    # Output: last position → sigmoid
    raw = h[-1] @ weights.W_out  # shape (1,) or scalar
    logit = float(np.squeeze(raw)) + weights.b_out
    return float(sigmoid(np.array([logit]))[0])


# ── Feature preparation ─────────────────────────────────────────────────


def prepare_features(df: pd.DataFrame, seq_len: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare (samples, seq_len, n_features) windows and binary targets.

    Expected columns: open, high, low, close, volume, vix (optional).
    Target: 1 if next-day close > today's close.
    """
    close = df["close"].values.astype(float)
    features_list = []

    # Returns
    returns = np.zeros(len(close))
    returns[1:] = (close[1:] - close[:-1]) / close[:-1]
    features_list.append(returns)

    # Normalised OHLC
    for col in ["open", "high", "low"]:
        if col in df.columns:
            features_list.append(df[col].values / close - 1)

    # Volume (normalised)
    if "volume" in df.columns:
        vol = df["volume"].values.astype(float)
        vol_ma = pd.Series(vol).rolling(20).mean().fillna(vol.mean()).values
        features_list.append(vol / np.maximum(vol_ma, 1) - 1)

    # VIX (normalised)
    if "vix" in df.columns:
        vix = df["vix"].values.astype(float)
        features_list.append((vix - 20) / 10)  # centre around 20

    # Momentum
    features_list.append(pd.Series(returns).rolling(5).sum().fillna(0).values)
    features_list.append(pd.Series(returns).rolling(10).sum().fillna(0).values)

    # Pad to 8 features
    while len(features_list) < 8:
        features_list.append(np.zeros(len(close)))

    raw = np.column_stack(features_list[:8])

    # Build windows
    X, y = [], []
    for i in range(seq_len, len(close) - 1):
        X.append(raw[i - seq_len:i])
        y.append(1.0 if close[i + 1] > close[i] else 0.0)

    return np.array(X), np.array(y)


# ── Training (evolutionary search) ──────────────────────────────────────


def evaluate_weights(
    weights: TransformerWeights,
    cfg: TransformerConfig,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """Evaluate accuracy and loss of weights on data."""
    n = len(X)
    correct = 0
    total_loss = 0.0

    for i in range(n):
        prob = transformer_forward(X[i], weights, cfg)
        pred = 1 if prob > 0.5 else 0
        if pred == int(y[i]):
            correct += 1
        # Binary cross-entropy
        eps = 1e-7
        loss = -(y[i] * math.log(prob + eps) + (1 - y[i]) * math.log(1 - prob + eps))
        total_loss += loss

    accuracy = correct / n if n > 0 else 0
    avg_loss = total_loss / n if n > 0 else 0
    return accuracy, avg_loss


def perturb_weights(
    weights: TransformerWeights,
    cfg: TransformerConfig,
    scale: float = 0.01,
    seed: int = 0,
) -> TransformerWeights:
    """Add random perturbation to weights."""
    rng = np.random.RandomState(seed)
    w = TransformerWeights()
    w.W_in = weights.W_in + rng.randn(*weights.W_in.shape) * scale
    w.W_q = [q + rng.randn(*q.shape) * scale for q in weights.W_q]
    w.W_k = [k + rng.randn(*k.shape) * scale for k in weights.W_k]
    w.W_v = [v + rng.randn(*v.shape) * scale for v in weights.W_v]
    w.W_o = [o + rng.randn(*o.shape) * scale for o in weights.W_o]
    w.W_ff1 = [f + rng.randn(*f.shape) * scale for f in weights.W_ff1]
    w.W_ff2 = [f + rng.randn(*f.shape) * scale for f in weights.W_ff2]
    w.W_out = weights.W_out + rng.randn(*weights.W_out.shape) * scale
    w.b_out = weights.b_out + rng.randn() * scale
    return w


def train_evolutionary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: TransformerConfig,
    n_iterations: int = 20,
    population: int = 5,
    seed: int = 42,
) -> TransformerWeights:
    """Train via evolutionary strategy (random search with perturbation)."""
    best_weights = init_weights(cfg, seed)
    # Subsample for speed
    n = min(len(X_train), 100)
    idx = np.random.RandomState(seed).choice(len(X_train), n, replace=False)
    X_sub, y_sub = X_train[idx], y_train[idx]

    best_acc, _ = evaluate_weights(best_weights, cfg, X_sub, y_sub)

    for it in range(n_iterations):
        for p in range(population):
            candidate = perturb_weights(best_weights, cfg, scale=0.02, seed=seed + it * population + p)
            acc, _ = evaluate_weights(candidate, cfg, X_sub, y_sub)
            if acc > best_acc:
                best_acc = acc
                best_weights = candidate

    return best_weights


# ── Walk-forward evaluation ──────────────────────────────────────────────


@dataclass
class FoldResult:
    fold: int
    train_acc: float
    test_acc: float
    test_sharpe: float
    n_train: int
    n_test: int


@dataclass
class ComparisonResult:
    """Transformer vs XGBoost comparison."""

    transformer_acc: float
    transformer_sharpe: float
    xgb_acc: float
    xgb_sharpe: float
    transformer_wins: bool


@dataclass
class PredictorResult:
    """Full evaluation result."""

    config: TransformerConfig
    folds: List[FoldResult]
    avg_oos_accuracy: float
    avg_oos_sharpe: float
    comparison: ComparisonResult
    n_observations: int


def compute_signal_sharpe(probs: np.ndarray, returns: np.ndarray) -> float:
    """Sharpe of going long when P(up)>0.5, short when P(up)<0.5."""
    pnls = []
    for i in range(len(probs)):
        if probs[i] > 0.55:
            pnls.append(returns[i])
        elif probs[i] < 0.45:
            pnls.append(-returns[i])
    if len(pnls) < 5:
        return 0.0
    arr = np.array(pnls)
    mu, std = arr.mean(), arr.std(ddof=1)
    return float(mu / std * math.sqrt(TRADING_DAYS)) if std > 1e-12 else 0.0


# ── Core predictor ───────────────────────────────────────────────────────


class TransformerPredictor:
    """Lightweight transformer for direction prediction."""

    def __init__(self, config: Optional[TransformerConfig] = None):
        self.config = config or TransformerConfig()

    def predict(self, X: np.ndarray, weights: TransformerWeights) -> np.ndarray:
        """Predict P(up) for each sample."""
        probs = np.zeros(len(X))
        for i in range(len(X)):
            probs[i] = transformer_forward(X[i], weights, self.config)
        return probs

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        n_folds: int = 3,
        train_years: int = 3,
        n_iterations: int = 15,
    ) -> PredictorResult:
        """Walk-forward train and evaluate."""
        cfg = self.config
        X, y = prepare_features(df, cfg.seq_len)

        if len(X) < 100:
            return PredictorResult(cfg, [], 0.5, 0.0,
                                    ComparisonResult(0.5, 0, 0.5, 0, False), len(df))

        # Returns for Sharpe computation
        close = df["close"].values
        returns = np.zeros(len(close))
        returns[1:] = (close[1:] - close[:-1]) / close[:-1]
        test_returns = returns[cfg.seq_len + 1:]  # aligned with y

        n = len(X)
        fold_size = n // (n_folds + 1)
        folds: List[FoldResult] = []
        all_test_probs = []
        all_test_y = []
        all_test_rets = []

        for f in range(n_folds):
            train_end = fold_size * (f + 1)
            test_start = train_end
            test_end = min(train_end + fold_size, n)
            if test_end <= test_start:
                continue

            X_tr, y_tr = X[:train_end], y[:train_end]
            X_te, y_te = X[test_start:test_end], y[test_start:test_end]

            weights = train_evolutionary(X_tr, y_tr, cfg, n_iterations, seed=42 + f)

            train_acc, _ = evaluate_weights(weights, cfg, X_tr[:50], y_tr[:50])
            test_probs = self.predict(X_te, weights)
            test_pred = (test_probs > 0.5).astype(int)
            test_acc = float((test_pred == y_te.astype(int)).mean())

            test_rets = test_returns[test_start:test_end] if test_end <= len(test_returns) else np.zeros(test_end - test_start)
            test_sharpe = compute_signal_sharpe(test_probs, test_rets[:len(test_probs)])

            folds.append(FoldResult(f + 1, train_acc, test_acc, test_sharpe, train_end, test_end - test_start))
            all_test_probs.extend(test_probs)
            all_test_y.extend(y_te)
            all_test_rets.extend(test_rets[:len(test_probs)])

        avg_acc = float(np.mean([f.test_acc for f in folds])) if folds else 0.5
        avg_sharpe = float(np.mean([f.test_sharpe for f in folds])) if folds else 0.0

        # XGBoost comparison
        comparison = self._xgb_comparison(X, y, test_returns, n_folds, fold_size)

        return PredictorResult(
            config=cfg, folds=folds,
            avg_oos_accuracy=avg_acc, avg_oos_sharpe=avg_sharpe,
            comparison=comparison, n_observations=len(df),
        )

    def _xgb_comparison(
        self, X: np.ndarray, y: np.ndarray, returns: np.ndarray,
        n_folds: int, fold_size: int,
    ) -> ComparisonResult:
        """Run XGBoost baseline for comparison."""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return ComparisonResult(0.5, 0, 0.5, 0, False)

        n = len(X)
        X_flat = X.reshape(n, -1)  # flatten sequence
        accs, sharpes = [], []

        for f in range(n_folds):
            train_end = fold_size * (f + 1)
            test_start = train_end
            test_end = min(train_end + fold_size, n)
            if test_end <= test_start:
                continue

            model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42, verbosity=0)
            model.fit(X_flat[:train_end], y[:train_end])
            probs = model.predict_proba(X_flat[test_start:test_end])[:, 1]
            preds = (probs > 0.5).astype(int)
            acc = float((preds == y[test_start:test_end].astype(int)).mean())
            accs.append(acc)

            test_rets = returns[test_start:test_end] if test_end <= len(returns) else np.zeros(test_end - test_start)
            sharpes.append(compute_signal_sharpe(probs, test_rets[:len(probs)]))

        xgb_acc = float(np.mean(accs)) if accs else 0.5
        xgb_sharpe = float(np.mean(sharpes)) if sharpes else 0.0

        # Transformer results from caller
        return ComparisonResult(0, 0, xgb_acc, xgb_sharpe, False)

    @staticmethod
    def generate_report(result: PredictorResult, output_path: Path = DEFAULT_OUTPUT) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_build_html(result), encoding="utf-8")
        return output_path


def _fr(v): return f"{v:.3f}"
def _fp(v): return f"{v:.1%}"


def _build_html(r: PredictorResult) -> str:
    c = r.comparison
    fold_rows = "".join(
        f"<tr><td>{f.fold}</td><td>{f.n_train}</td><td>{f.n_test}</td>"
        f"<td>{_fp(f.train_acc)}</td><td>{_fp(f.test_acc)}</td><td>{_fr(f.test_sharpe)}</td></tr>"
        for f in r.folds
    )
    winner = "Transformer" if r.avg_oos_accuracy > c.xgb_acc else "XGBoost"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Transformer Predictor</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:900px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}h1,h2{{color:#58a6ff}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:20px 0}}.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;text-align:center}}.c .l{{color:#8b949e;font-size:.8em}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1.1em}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}}th{{color:#8b949e;background:#161b22}}</style></head><body>
<h1>Transformer Price Predictor</h1>
<div class="cards">
<div class="c"><div class="l">Transformer OOS Acc</div><div class="v">{_fp(r.avg_oos_accuracy)}</div></div>
<div class="c"><div class="l">Transformer Sharpe</div><div class="v">{_fr(r.avg_oos_sharpe)}</div></div>
<div class="c"><div class="l">XGBoost OOS Acc</div><div class="v">{_fp(c.xgb_acc)}</div></div>
<div class="c"><div class="l">XGBoost Sharpe</div><div class="v">{_fr(c.xgb_sharpe)}</div></div>
<div class="c"><div class="l">Winner</div><div class="v">{winner}</div></div>
<div class="c"><div class="l">Architecture</div><div class="v">{r.config.n_layers}L {r.config.d_model}D {r.config.n_heads}H</div></div>
</div>
<h2>Walk-Forward Folds</h2>
<table><tr><th>Fold</th><th>Train N</th><th>Test N</th><th>Train Acc</th><th>Test Acc</th><th>Test Sharpe</th></tr>{fold_rows}</table>
</body></html>"""
