"""
Forex Trading Environment for Deep Reinforcement Learning
==========================================================

This module implements a custom Gymnasium environment for training PPO agents
on Forex trading with a focus on prop firm compliance and risk management.

Key Features:
- State space: Normalized technical indicators + position/account state (NO raw prices)
- Action space: [position_direction, risk_percentage] for risk-based position sizing
- Safety layer: Hard guardrails for prop firm limits
- Reward: Sortino-based with heavy drawdown penalties

Author: DRL Trading System
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from collections import deque
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.environment.preprocessing import ForexPreprocessor
from src.utils.reward_calculator import RewardCalculator
from src.agents.safety_layer import PropFirmSafetyLayer


class ForexTradingEnv(gym.Env):
    """
    Custom Gymnasium environment for Forex trading with prop firm constraints.

    This environment simulates trading a single currency pair (e.g., EURUSD) with
    risk-based position sizing and strict safety guardrails.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self,
                 data: pd.DataFrame,
                 initial_balance: float = 100000.0,
                 max_episode_steps: int = 1000,
                 transaction_cost_pct: float = 0.0001,  # 1 pip spread
                 leverage: int = 30,
                 max_risk_per_trade: float = 0.02,  # 2% max per trade
                 atr_multiplier_for_sl: float = 2.0,  # Stop loss = 2x ATR
                 daily_loss_limit: float = 0.04,
                 max_drawdown_limit: float = 0.10,
                 returns_history_len: int = 50,
                 render_mode: Optional[str] = None):
        """
        Initialize the trading environment.

        Args:
            data: Preprocessed DataFrame with features (NO raw prices)
            initial_balance: Starting account balance
            max_episode_steps: Maximum steps per episode
            transaction_cost_pct: Transaction cost (spread) as percentage
            leverage: Leverage ratio
            max_risk_per_trade: Maximum risk per trade (as decimal)
            atr_multiplier_for_sl: ATR multiplier for stop-loss calculation
            daily_loss_limit: Daily loss limit for prop firm
            max_drawdown_limit: Maximum drawdown limit
            returns_history_len: Length of returns history for reward calculation
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_episode_steps = max_episode_steps
        self.transaction_cost_pct = transaction_cost_pct
        self.leverage = leverage
        self.max_risk_per_trade = max_risk_per_trade
        self.atr_multiplier_for_sl = atr_multiplier_for_sl
        self.render_mode = render_mode
        self.returns_history_len = returns_history_len

        # Validate data
        if len(self.data) < max_episode_steps:
            raise ValueError(f"Data length ({len(self.data)}) must be >= max_episode_steps ({max_episode_steps})")

        # Number of features from preprocessed data
        self.n_features = len(self.data.columns)

        # Additional state features: position, unrealized_pnl, time_in_position, drawdown, daily_pnl
        self.n_state_features = self.n_features + 5

        # ====================================================================
        # ACTION SPACE
        # ====================================================================
        # Action is 2D continuous:
        #   action[0]: Position direction (-1 = full short, 0 = close, +1 = full long)
        #   action[1]: Risk percentage for this trade (0 to 1, maps to 0% to 2% account risk)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # ====================================================================
        # OBSERVATION SPACE
        # ====================================================================
        # State includes normalized features + position state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_state_features,),
            dtype=np.float32
        )

        # ====================================================================
        # INITIALIZE COMPONENTS
        # ====================================================================
        self.reward_calculator = RewardCalculator()
        self.safety_layer = PropFirmSafetyLayer(
            daily_loss_limit=daily_loss_limit,
            max_drawdown_limit=max_drawdown_limit,
            enable_logging=False  # Disable during training for cleaner logs
        )

        # ====================================================================
        # EPISODE STATE VARIABLES (reset in reset())
        # ====================================================================
        self.current_step = 0
        self.balance = initial_balance
        self.equity = initial_balance
        self.position = 0.0  # Current position size (lots)
        self.entry_price = 0.0
        self.position_value = 0.0
        self.unrealized_pnl = 0.0

        # Tracking
        self.returns_history = deque(maxlen=returns_history_len)
        self.peak_equity = initial_balance
        self.daily_start_balance = initial_balance
        self.trade_count = 0
        self.episode_trades = []

        # Episode start index (randomized for diverse training)
        self.start_index = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options

        Returns:
            Tuple of (initial_observation, info_dict)
        """
        super().reset(seed=seed)

        # Randomize starting point (but ensure enough data for episode)
        max_start = len(self.data) - self.max_episode_steps - 1
        self.start_index = self.np_random.integers(0, max_start) if max_start > 0 else 0

        # Reset state
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.position_value = 0.0
        self.unrealized_pnl = 0.0

        # Reset tracking
        self.returns_history.clear()
        self.peak_equity = self.initial_balance
        self.daily_start_balance = self.initial_balance
        self.trade_count = 0
        self.episode_trades = []

        # Reset safety layer
        self.safety_layer.reset(self.initial_balance)

        # Get initial observation
        obs = self._get_observation()

        info = self._get_info()

        return obs, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state).

        Returns:
            State vector with normalized features + position state
        """
        # Get current row of preprocessed features
        idx = self.start_index + self.current_step
        features = self.data.iloc[idx].values.astype(np.float32)

        # Add position state features
        position_state = np.array([
            np.clip(self.position, -1.0, 1.0),  # Normalized position size
            self.unrealized_pnl / self.balance if self.balance > 0 else 0.0,  # Unrealized P&L %
            min(self.current_step / 100.0, 1.0),  # Time in episode (normalized)
            self._calculate_drawdown(),  # Current drawdown %
            (self.balance - self.daily_start_balance) / self.daily_start_balance  # Daily P&L %
        ], dtype=np.float32)

        # Concatenate features and position state
        observation = np.concatenate([features, position_state])

        return observation

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    def _get_current_price(self) -> float:
        """
        Get current price for position valuation.

        Note: We need the close price from the ORIGINAL data (before preprocessing).
        This requires passing the original data or reconstructing from log returns.
        For simplicity in this implementation, we'll store a reference price.
        """
        # This is a simplified approach - in production, you'd store original close prices
        # For now, we'll use a proxy based on cumulative returns
        idx = self.start_index + self.current_step

        # Get log returns and reconstruct price (simplified)
        # In practice, you should pass original close prices separately
        if 'log_return_1' in self.data.columns:
            # Use cumulative returns to reconstruct relative price
            # This is approximate - better to store original prices
            log_returns = self.data['log_return_1'].iloc[self.start_index:idx+1]
            cumulative_return = np.exp(log_returns.sum())
            # Use a reference price (e.g., EURUSD around 1.0950)
            reference_price = 1.0950
            return reference_price * cumulative_return
        else:
            # Fallback: use a constant price (not ideal but prevents errors)
            return 1.0950

    def _calculate_position_size(self, risk_pct: float, current_price: float) -> float:
        """
        Calculate position size based on risk percentage and ATR.

        Risk-based position sizing:
        Position Size = (Account Risk) / (Stop Loss Distance)

        Args:
            risk_pct: Risk percentage (0 to 1, maps to 0% to max_risk_per_trade)
            current_price: Current market price

        Returns:
            Position size in lots
        """
        # Map risk_pct to actual account risk
        account_risk = risk_pct * self.max_risk_per_trade * self.balance

        # Get ATR for stop-loss calculation
        idx = self.start_index + self.current_step
        if 'atr_normalized' in self.data.columns:
            atr_normalized = self.data.iloc[idx]['atr_normalized']
            atr = atr_normalized * current_price  # Denormalize
        else:
            atr = current_price * 0.001  # Fallback: 0.1% of price

        # Stop loss distance
        stop_loss_distance = self.atr_multiplier_for_sl * atr

        # Position size
        if stop_loss_distance > 0:
            position_size = account_risk / stop_loss_distance
        else:
            position_size = 0.0

        # Apply leverage constraint (position value <= balance * leverage)
        max_position_value = self.balance * self.leverage
        max_position_lots = max_position_value / (current_price * 100000)  # Assuming standard lot = 100k

        position_size = min(position_size, max_position_lots)

        return position_size

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step.

        Args:
            action: [position_direction, risk_pct]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # ====================================================================
        # 1. SAFETY LAYER CHECK (before execution)
        # ====================================================================
        action, was_overridden = self.safety_layer.check_and_override_action(
            action=action,
            current_balance=self.balance,
            current_position=self.position,
            unrealized_pnl=self.unrealized_pnl
        )

        # ====================================================================
        # 2. PARSE ACTION
        # ====================================================================
        position_direction = np.clip(action[0], -1.0, 1.0)  # -1 to +1
        risk_pct = np.clip(action[1], 0.0, 1.0)  # 0 to 1

        # ====================================================================
        # 3. GET CURRENT PRICE
        # ====================================================================
        current_price = self._get_current_price()

        # ====================================================================
        # 4. EXECUTE TRADE
        # ====================================================================
        old_position = self.position
        position_changed = False

        # Close threshold: consider action as "close" if direction is near zero
        if np.abs(position_direction) < 0.1:
            # Close position
            if self.position != 0:
                self._close_position(current_price)
                position_changed = True
        else:
            # Calculate new position size
            desired_position_size = self._calculate_position_size(risk_pct, current_price)
            desired_position = np.sign(position_direction) * desired_position_size

            # Check if we're changing position
            if np.abs(desired_position - self.position) > 1e-6:
                # Close old position if it exists
                if self.position != 0:
                    self._close_position(current_price)

                # Open new position
                self._open_position(desired_position, current_price)
                position_changed = True

        # ====================================================================
        # 5. UPDATE UNREALIZED P&L
        # ====================================================================
        if self.position != 0:
            # Calculate unrealized P&L
            price_change = current_price - self.entry_price
            pnl_per_lot = price_change * 100000 * np.sign(self.position)  # Standard lot
            self.unrealized_pnl = pnl_per_lot * np.abs(self.position)
            self.equity = self.balance + self.unrealized_pnl
        else:
            self.unrealized_pnl = 0.0
            self.equity = self.balance

        # ====================================================================
        # 6. UPDATE PEAK EQUITY & TRACKING
        # ====================================================================
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Calculate step return
        step_return_pct = (self.equity / self.daily_start_balance - 1.0)
        self.returns_history.append(step_return_pct)

        # ====================================================================
        # 7. CALCULATE REWARD
        # ====================================================================
        current_drawdown = self._calculate_drawdown()
        step_pnl_pct = (self.equity - self.balance) / self.balance if self.balance > 0 else 0.0

        reward = self.reward_calculator.calculate_reward(
            returns_history=np.array(self.returns_history),
            current_step_pnl_pct=step_pnl_pct,
            current_drawdown_pct=current_drawdown,
            position_changed=position_changed
        )

        # ====================================================================
        # 8. INCREMENT STEP
        # ====================================================================
        self.current_step += 1

        # ====================================================================
        # 9. CHECK TERMINATION CONDITIONS
        # ====================================================================
        terminated = False
        truncated = False

        # Check if episode should end
        if self.current_step >= self.max_episode_steps:
            truncated = True

        # Check if out of data
        if self.start_index + self.current_step >= len(self.data):
            truncated = True

        # Check if daily loss limit breached
        daily_loss_pct = self.safety_layer.calculate_daily_loss_pct(self.balance)
        if daily_loss_pct >= self.safety_layer.daily_loss_limit:
            terminated = True

        # Check if max drawdown breached
        if current_drawdown >= self.safety_layer.max_drawdown_limit:
            terminated = True

        # Check if account blown (balance <= 0)
        if self.balance <= 0:
            terminated = True

        # Close any open positions at episode end
        if terminated or truncated:
            if self.position != 0:
                self._close_position(current_price)

        # ====================================================================
        # 10. GET OBSERVATION & INFO
        # ====================================================================
        observation = self._get_observation()
        info = self._get_info()
        info['was_action_overridden'] = was_overridden

        return observation, reward, terminated, truncated, info

    def _open_position(self, position_size: float, entry_price: float):
        """Open a new position."""
        self.position = position_size
        self.entry_price = entry_price
        self.position_value = np.abs(position_size) * entry_price * 100000

        # Apply transaction cost
        transaction_cost = self.position_value * self.transaction_cost_pct
        self.balance -= transaction_cost

        self.trade_count += 1
        self.episode_trades.append({
            'step': self.current_step,
            'action': 'OPEN',
            'position': position_size,
            'price': entry_price,
            'balance': self.balance
        })

    def _close_position(self, exit_price: float):
        """Close current position and realize P&L."""
        if self.position == 0:
            return

        # Calculate realized P&L
        price_change = exit_price - self.entry_price
        pnl_per_lot = price_change * 100000 * np.sign(self.position)
        realized_pnl = pnl_per_lot * np.abs(self.position)

        # Apply transaction cost
        transaction_cost = self.position_value * self.transaction_cost_pct

        # Update balance
        self.balance += realized_pnl - transaction_cost
        self.equity = self.balance

        self.episode_trades.append({
            'step': self.current_step,
            'action': 'CLOSE',
            'position': 0.0,
            'price': exit_price,
            'pnl': realized_pnl,
            'balance': self.balance
        })

        # Reset position
        self.position = 0.0
        self.entry_price = 0.0
        self.position_value = 0.0
        self.unrealized_pnl = 0.0

    def _get_info(self) -> dict:
        """Get info dictionary."""
        return {
            'step': self.current_step,
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'drawdown': self._calculate_drawdown(),
            'peak_equity': self.peak_equity,
            'trade_count': self.trade_count,
            'total_return': (self.equity / self.initial_balance) - 1.0
        }

    def render(self):
        """Render environment (optional)."""
        if self.render_mode == 'human':
            print(f"Step: {self.current_step} | "
                  f"Balance: ${self.balance:,.2f} | "
                  f"Equity: ${self.equity:,.2f} | "
                  f"Position: {self.position:.2f} | "
                  f"Drawdown: {self._calculate_drawdown():.2%}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the ForexTradingEnv.
    """
    from src.utils.data_loader import ForexDataLoader
    from src.environment.preprocessing import ForexPreprocessor

    print("=" * 80)
    print("FOREX TRADING ENVIRONMENT - TEST MODE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Load and preprocess data
    # -------------------------------------------------------------------------
    print("\n[1] Creating sample data...")
    loader = ForexDataLoader()
    sample_path = "data/EURUSD_15m_sample.csv"

    # Create sample data if it doesn't exist
    if not os.path.exists(sample_path):
        loader.create_sample_data(sample_path, num_rows=5000, timeframe='15min')

    print("\n[2] Loading data...")
    raw_data = loader.load_forex_data(sample_path, validate=True)

    print("\n[3] Preprocessing data...")
    preprocessor = ForexPreprocessor()
    processed_data = preprocessor.process_dataframe(raw_data)

    print(f"\nProcessed data shape: {processed_data.shape}")
    print(f"Features: {processed_data.columns.tolist()}")

    # -------------------------------------------------------------------------
    # 2. Create environment
    # -------------------------------------------------------------------------
    print("\n[4] Creating trading environment...")
    env = ForexTradingEnv(
        data=processed_data,
        initial_balance=100000.0,
        max_episode_steps=1000,
        render_mode='human'
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # -------------------------------------------------------------------------
    # 3. Test with random actions
    # -------------------------------------------------------------------------
    print("\n[5] Testing with random actions (10 steps)...")
    obs, info = env.reset()

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Balance: ${info['balance']:,.2f}")
        print(f"  Equity: ${info['equity']:,.2f}")
        print(f"  Position: {info['position']:.4f}")

        if terminated or truncated:
            print(f"  Episode ended ({'terminated' if terminated else 'truncated'})")
            break

    print("\n" + "=" * 80)
    print("ENVIRONMENT TEST COMPLETE")
    print("=" * 80)
    print("\nThe environment is ready for PPO training!")
    print("Next step: Run train_ppo.py to train the agent")
    print("=" * 80)
