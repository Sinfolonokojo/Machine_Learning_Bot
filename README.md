# Deep Reinforcement Learning Forex Trading Agent

A robust, low-risk trading agent designed for Forex prop firm challenges using Deep Reinforcement Learning (PPO) with strict risk management and safety guardrails.

## Overview

This project implements a **capital-preserving** trading agent specifically optimized for passing proprietary trading firm evaluations. The agent prioritizes:

- **Risk-first approach**: Sortino ratio-based rewards with heavy drawdown penalties
- **Safety guardrails**: Hard stops at daily loss and drawdown limits
- **Non-stationary data handling**: Log returns and normalized features (no raw prices)
- **Professional risk management**: Risk-based position sizing with ATR-based stop losses

## Architecture

### Core Components

```
Machine_Learning_Bot/
├── src/
│   ├── environment/
│   │   ├── trading_env.py          # Custom Gymnasium environment
│   │   └── preprocessing.py         # Feature engineering pipeline
│   ├── agents/
│   │   └── safety_layer.py          # Hard risk guardrails
│   ├── training/
│   │   └── train_ppo.py             # PPO training script
│   └── utils/
│       ├── reward_calculator.py     # Sortino-based reward function
│       └── data_loader.py           # CSV loading utilities
├── data/                            # Place your CSV files here
├── models/                          # Saved models
├── logs/                            # Tensorboard logs
└── requirements.txt
```

## Key Features

### 1. State Space (20-25 dimensions)
- **Log returns**: 1, 3, 5 period returns (stationary)
- **Technical indicators** (normalized):
  - RSI (0-1 range)
  - SMA/EMA distances (ATR-normalized)
  - MACD (line, signal, histogram)
  - Bollinger Bands (%B position)
  - Stochastic oscillator (%K, %D)
  - ATR (volatility measure)
- **Position state**: Current position, unrealized P&L, time in position
- **Account state**: Drawdown, daily P&L

### 2. Action Space (2D continuous)
- `action[0]`: Position direction (-1 = full short, 0 = close, +1 = full long)
- `action[1]`: Risk percentage (0-1, maps to 0-2% account risk)

### 3. Reward Function (Risk-Averse)

```python
reward = (
    sortino_ratio * 5.0 +          # Primary: risk-adjusted returns
    step_pnl * 0.5 +                # Secondary: actual profit
    -10.0 * drawdown²  +            # Critical: quadratic drawdown penalty
    -2.0 * volatility +             # Prefer stable equity
    -0.1 * position_changes +       # Discourage overtrading
    consistency_bonus               # Reward steady growth
)
```

**Key properties**:
- Sortino ratio weighted 5x more than raw P&L
- Drawdown penalty is quadratic (exponentially punishes large losses)
- Discourages overtrading and high volatility

### 4. Safety Layer (Hard Guardrails)

The safety layer overrides agent actions when approaching risk limits:

- **Daily loss limit**: 4% (hard stop at 3.5%)
- **Max drawdown**: 10% (hard stop at 8%)
- **Risk reduction zone**: Reduces position sizes when approaching limits
- **Position size caps**: Maximum position constraints

**CRITICAL**: This layer operates OUTSIDE the neural network to prevent reward exploitation.

## Installation

### Prerequisites
- Python 3.9+
- TA-Lib C library (see installation notes below)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd Machine_Learning_Bot
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### TA-Lib Installation

**Windows**:
```bash
# Download TA-Lib wheel from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.XX-cpXX-cpXX-win_amd64.whl
```

**Linux/Mac**:
```bash
# Install TA-Lib C library first
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Then install Python wrapper
pip install TA-Lib
```

**Alternative**: If TA-Lib installation fails, use `pandas-ta` instead (see requirements.txt).

## Usage

### Quick Start (with sample data)

```bash
# Train the agent with sample data
python src/training/train_ppo.py
```

This will:
1. Generate sample EURUSD 15-minute data
2. Preprocess features
3. Train PPO agent for 500,000 steps
4. Evaluate on test set
5. Save model to `models/ppo_forex_v1.zip`

### Training with Real Data

1. **Prepare your data**: Place CSV file in `data/` folder

   **Required format**:
   ```csv
   timestamp,open,high,low,close,volume
   2024-01-01 00:00:00,1.0950,1.0955,1.0948,1.0952,1234
   2024-01-01 00:15:00,1.0952,1.0957,1.0951,1.0956,1456
   ...
   ```

2. **Update training script**:
   ```python
   # In src/training/train_ppo.py, line 264:
   DATA_PATH = "data/EURUSD_15m.csv"  # <-- Your real data file
   ```

3. **Run training**:
   ```bash
   python src/training/train_ppo.py
   ```

### Monitor Training

```bash
# View training progress in Tensorboard
tensorboard --logdir=logs/
```

Open http://localhost:6006 in your browser.

### Load Trained Model

```python
from stable_baselines3 import PPO

# Load model
model = PPO.load("models/ppo_forex_v1")

# Make predictions
obs = env.reset()
action, _states = model.predict(obs, deterministic=True)
```

## Configuration

### Hyperparameters (in `train_ppo.py`)

```python
model = PPO(
    learning_rate=3e-5,          # Conservative learning rate
    n_steps=2048,                # Long rollouts for stable learning
    batch_size=64,
    n_epochs=10,
    gamma=0.99,                  # Long-term focus
    clip_range=0.2,              # Standard PPO clip
    ent_coef=0.01,               # Exploration coefficient
)
```

### Environment Settings (in `trading_env.py`)

```python
env = ForexTradingEnv(
    initial_balance=100000.0,
    max_episode_steps=1000,      # ~10 days of 15min candles
    transaction_cost_pct=0.0001, # 1 pip spread
    leverage=30,
    max_risk_per_trade=0.02,     # 2% max risk per trade
    daily_loss_limit=0.04,       # 4% daily loss limit
    max_drawdown_limit=0.10,     # 10% max drawdown
)
```

### Reward Weights (in `reward_calculator.py`)

```python
calculator = RewardCalculator(
    sortino_weight=5.0,          # Primary driver
    pnl_weight=0.5,              # Secondary
    drawdown_weight=10.0,        # Heavy penalty
    volatility_weight=2.0,
    transaction_weight=0.1,
    consistency_weight=2.0,
)
```

## Testing Individual Components

Each module can be tested independently:

```bash
# Test preprocessing pipeline
python src/environment/preprocessing.py

# Test data loader
python src/utils/data_loader.py

# Test reward calculator
python src/utils/reward_calculator.py

# Test safety layer
python src/agents/safety_layer.py

# Test trading environment
python src/environment/trading_env.py
```

## Expected Performance

After training, the agent should exhibit:

- **Win rate**: 45-55% (slightly better than random)
- **Profit factor**: 1.3-1.8
- **Max drawdown**: < 5%
- **Sortino ratio**: > 1.5
- **Avg trade duration**: 4-12 hours (16-48 candles)
- **Trades per week**: 5-15 (not overtrading)

## Project Timeline

### Phase 1: Development & Testing (Current)
- ✅ Core infrastructure implemented
- ✅ All components with test modes
- ⏳ Training with sample data

### Phase 2: Real Data Training
- Load 2+ years of EURUSD 15-minute data
- Train for 1-5 million timesteps
- Hyperparameter tuning

### Phase 3: Validation
- Walk-forward validation
- Out-of-sample testing
- Safety layer validation

### Phase 4: Deployment
- Paper trading integration
- Live monitoring dashboard
- Performance tracking

## Troubleshooting

### Issue: TA-Lib installation fails
**Solution**: Use `pandas-ta` instead (uncomment in requirements.txt)

### Issue: Out of memory during training
**Solution**:
- Reduce `n_steps` in PPO config
- Use smaller network architecture
- Reduce `max_episode_steps`

### Issue: Agent not learning
**Solution**:
- Check reward function weights
- Increase `total_timesteps`
- Verify data quality
- Reduce `learning_rate`

### Issue: Safety layer triggers too often
**Solution**:
- Adjust `daily_loss_buffer` and `drawdown_buffer`
- Check if reward function is too aggressive
- Verify transaction costs aren't too high

## Contributing

This is a research project. Contributions welcome for:
- Additional technical indicators
- Alternative reward functions
- Hyperparameter optimization
- Documentation improvements

## Disclaimer

**IMPORTANT**: This is an educational/research project.

- **Not financial advice**: Do not use for live trading without extensive testing
- **No guarantees**: Past performance does not indicate future results
- **Risk warning**: Forex trading carries significant risk of loss
- **Prop firm rules**: Verify compliance with specific prop firm requirements

Always paper trade extensively before considering live deployment.

## License

MIT License - See LICENSE file for details

## Contact

For questions or collaboration: [Your contact information]

---

**Built with**: Python, Stable-Baselines3, Gymnasium, PyTorch

**Designed for**: Low-risk prop firm trading challenges

**Focus**: Capital preservation over high returns
