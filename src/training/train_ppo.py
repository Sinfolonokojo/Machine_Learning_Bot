"""
PPO Training Script for Forex Trading Agent
===========================================

This script trains a PPO agent using stable-baselines3 for low-risk Forex trading.

Training Strategy:
1. Load and preprocess historical data
2. Create trading environment with safety guardrails
3. Train PPO with conservative hyperparameters
4. Save model checkpoints and logs
5. Evaluate performance on validation set

Author: DRL Trading System
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.environment.trading_env import ForexTradingEnv
from src.environment.preprocessing import ForexPreprocessor
from src.utils.data_loader import ForexDataLoader


class TradingMetricsCallback(BaseCallback):
    """
    Custom callback to log trading-specific metrics during training.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_drawdowns = []
        self.episode_trades = []

    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]

            # Log metrics
            self.episode_returns.append(info.get('total_return', 0.0))
            self.episode_drawdowns.append(info.get('drawdown', 0.0))
            self.episode_trades.append(info.get('trade_count', 0))

            # Log to tensorboard
            if len(self.episode_returns) % 10 == 0:  # Every 10 episodes
                self.logger.record('trading/mean_return', np.mean(self.episode_returns[-10:]))
                self.logger.record('trading/mean_drawdown', np.mean(self.episode_drawdowns[-10:]))
                self.logger.record('trading/mean_trades', np.mean(self.episode_trades[-10:]))

        return True


def create_env(data: pd.DataFrame, initial_balance: float = 100000.0) -> DummyVecEnv:
    """
    Create and wrap the trading environment.

    Args:
        data: Preprocessed trading data
        initial_balance: Starting account balance

    Returns:
        Vectorized environment
    """
    def make_env():
        env = ForexTradingEnv(
            data=data,
            initial_balance=initial_balance,
            max_episode_steps=500,  # Reduced to fit smaller datasets
            transaction_cost_pct=0.0001,  # 1 pip spread
            leverage=30,
            max_risk_per_trade=0.02,
            atr_multiplier_for_sl=2.0,
            daily_loss_limit=0.04,
            max_drawdown_limit=0.10,
            returns_history_len=50
        )
        # Wrap with Monitor for logging
        env = Monitor(env)
        return env

    # Create vectorized environment
    env = DummyVecEnv([make_env])

    # Note: VecNormalize can help with observation normalization
    # Uncomment if needed (may need tuning)
    # env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    return env


def train_ppo_agent(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    total_timesteps: int = 500_000,
    initial_balance: float = 100000.0,
    model_save_path: str = "models/ppo_forex_v1",
    log_dir: str = "logs/",
    checkpoint_freq: int = 50_000
):
    """
    Train PPO agent with conservative hyperparameters.

    Args:
        train_data: Preprocessed training data
        val_data: Preprocessed validation data
        total_timesteps: Total training timesteps
        initial_balance: Starting account balance
        model_save_path: Path to save final model
        log_dir: Directory for tensorboard logs
        checkpoint_freq: Checkpoint frequency (steps)

    Returns:
        Trained PPO model
    """
    print("=" * 80)
    print("PPO TRAINING - FOREX TRADING AGENT")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Create environments
    # -------------------------------------------------------------------------
    print("\n[1] Creating training environment...")
    train_env = create_env(train_data, initial_balance)

    print("[2] Creating validation environment...")
    val_env = create_env(val_data, initial_balance)

    # -------------------------------------------------------------------------
    # 2. Initialize PPO model
    # -------------------------------------------------------------------------
    print("\n[3] Initializing PPO model...")

    # PPO Hyperparameters (Conservative settings for stable training)
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-5,          # Low learning rate for stability
        n_steps=2048,                 # Longer rollouts (more experience per update)
        batch_size=64,                # Batch size for training
        n_epochs=10,                  # Number of epochs for each update
        gamma=0.99,                   # Discount factor (long-term focus)
        gae_lambda=0.95,              # GAE lambda for advantage estimation
        clip_range=0.2,               # PPO clip range
        ent_coef=0.01,                # Entropy coefficient (encourage exploration)
        vf_coef=0.5,                  # Value function coefficient
        max_grad_norm=0.5,            # Gradient clipping
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Neural network architecture
        )
    )

    print("\nModel configuration:")
    print(f"  Policy: MlpPolicy")
    print(f"  Learning rate: {model.learning_rate}")
    print(f"  Batch size: {model.batch_size}")
    print(f"  Network architecture: [256, 256] (actor), [256, 256] (critic)")

    # -------------------------------------------------------------------------
    # 3. Setup callbacks
    # -------------------------------------------------------------------------
    print("\n[4] Setting up training callbacks...")

    # Checkpoint callback (save model periodically)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path='./models/checkpoints/',
        name_prefix='ppo_forex'
    )

    # Evaluation callback (evaluate on validation set)
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path='./models/best/',
        log_path='./logs/eval/',
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Custom trading metrics callback
    metrics_callback = TradingMetricsCallback()

    # Combine callbacks
    callback = CallbackList([checkpoint_callback, eval_callback, metrics_callback])

    # -------------------------------------------------------------------------
    # 4. Train the model
    # -------------------------------------------------------------------------
    print("\n[5] Starting training...")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Checkpoint frequency: {checkpoint_freq:,}")
    print(f"  Tensorboard logs: {log_dir}")
    print("\n" + "=" * 80)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 5. Save final model
    # -------------------------------------------------------------------------
    print(f"\n[6] Saving final model to: {model_save_path}")
    model.save(model_save_path)

    # -------------------------------------------------------------------------
    # 6. Cleanup
    # -------------------------------------------------------------------------
    train_env.close()
    val_env.close()

    return model


def evaluate_agent(
    model: PPO,
    test_data: pd.DataFrame,
    n_episodes: int = 10,
    initial_balance: float = 100000.0,
    render: bool = False
):
    """
    Evaluate trained agent on test data.

    Args:
        model: Trained PPO model
        test_data: Preprocessed test data
        n_episodes: Number of evaluation episodes
        initial_balance: Starting account balance
        render: Whether to render episodes

    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 80)
    print("AGENT EVALUATION")
    print("=" * 80)

    env = create_env(test_data, initial_balance)

    episode_returns = []
    episode_drawdowns = []
    episode_trades = []
    episode_final_balances = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]

            if render:
                env.render()

        # Log episode results
        final_info = info[0]
        episode_returns.append(final_info['total_return'])
        episode_drawdowns.append(final_info['drawdown'])
        episode_trades.append(final_info['trade_count'])
        episode_final_balances.append(final_info['balance'])

        print(f"\nEpisode {episode + 1}/{n_episodes}:")
        print(f"  Total Return: {final_info['total_return']:.2%}")
        print(f"  Max Drawdown: {final_info['drawdown']:.2%}")
        print(f"  Trades: {final_info['trade_count']}")
        print(f"  Final Balance: ${final_info['balance']:,.2f}")

    env.close()

    # Calculate statistics
    results = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_drawdown': np.mean(episode_drawdowns),
        'max_drawdown': np.max(episode_drawdowns),
        'mean_trades': np.mean(episode_trades),
        'win_rate': np.sum(np.array(episode_returns) > 0) / n_episodes,
        'avg_final_balance': np.mean(episode_final_balances)
    }

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Mean Return: {results['mean_return']:.2%} ± {results['std_return']:.2%}")
    print(f"Mean Drawdown: {results['mean_drawdown']:.2%}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Avg Trades per Episode: {results['mean_trades']:.1f}")
    print(f"Avg Final Balance: ${results['avg_final_balance']:,.2f}")
    print("=" * 80)

    return results


def main():
    """
    Main training pipeline.
    """
    # -------------------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------------------
    # ========================================================================
    # IMPORTANT: SWAP WITH YOUR REAL DATA
    # ========================================================================
    # To use real Forex data:
    # 1. Place your CSV file in the data/ folder
    # 2. Ensure format: timestamp,open,high,low,close,volume
    # 3. Update DATA_PATH below
    # ========================================================================

    DATA_PATH = "data/EURUSD_15m_sample.csv"  # <-- CHANGE THIS to your real data path
    INITIAL_BALANCE = 100000.0
    TOTAL_TIMESTEPS = 500_000  # Increase for longer training (e.g., 1-5 million)

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 1: LOAD DATA")
    print("=" * 80)

    loader = ForexDataLoader()

    # Create sample data if file doesn't exist
    if not os.path.exists(DATA_PATH):
        print(f"\n[WARNING] Data file not found: {DATA_PATH}")
        print("[INFO] Creating sample data for testing...")
        loader.create_sample_data(DATA_PATH, num_rows=10000, timeframe='15min')
        print("[INFO] Sample data created. Replace with real data for production training!")

    print(f"\nLoading data from: {DATA_PATH}")
    raw_data = loader.load_forex_data(DATA_PATH, validate=True)

    # -------------------------------------------------------------------------
    # 2. Preprocess data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: PREPROCESS DATA")
    print("=" * 80)

    preprocessor = ForexPreprocessor()
    processed_data = preprocessor.process_dataframe(raw_data)

    print(f"\nProcessed shape: {processed_data.shape}")
    print(f"Features: {len(preprocessor.get_feature_names())}")

    # -------------------------------------------------------------------------
    # 3. Split data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: SPLIT DATA")
    print("=" * 80)

    train_data, val_data, test_data = loader.split_data(
        processed_data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # -------------------------------------------------------------------------
    # 4. Train agent
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: TRAIN PPO AGENT")
    print("=" * 80)

    model = train_ppo_agent(
        train_data=train_data,
        val_data=val_data,
        total_timesteps=TOTAL_TIMESTEPS,
        initial_balance=INITIAL_BALANCE,
        model_save_path="models/ppo_forex_v1",
        log_dir="logs/",
        checkpoint_freq=50_000
    )

    # -------------------------------------------------------------------------
    # 5. Evaluate agent
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: EVALUATE AGENT")
    print("=" * 80)

    results = evaluate_agent(
        model=model,
        test_data=test_data,
        n_episodes=10,
        initial_balance=INITIAL_BALANCE,
        render=False
    )

    # -------------------------------------------------------------------------
    # 6. Final summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. View training logs: tensorboard --logdir=logs/")
    print("2. Test model: Load with PPO.load('models/ppo_forex_v1')")
    print("3. Deploy: Use model.predict() for live trading (with safety layer!)")
    print("4. Fine-tune: Adjust hyperparameters and retrain if needed")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
