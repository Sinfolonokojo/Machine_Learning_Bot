"""
Reward Calculator for Low-Risk Trading Agent
============================================

This module implements a sophisticated reward function designed for prop firm
trading challenges. The reward heavily emphasizes:

1. Sortino Ratio (penalize downside volatility, not upside)
2. Drawdown avoidance (quadratic penalty)
3. Consistency (reward steady positive returns)
4. Low trading frequency (transaction cost penalty)

The goal is to create a risk-averse agent that preserves capital above all else.

Author: DRL Trading System
"""

import numpy as np
from typing import List, Optional
from collections import deque


class RewardCalculator:
    """
    Calculates rewards for the trading environment with a focus on risk-adjusted returns.

    This reward function is specifically tuned for proprietary trading firm challenges
    where drawdown limits and consistency are critical for passing evaluations.
    """

    def __init__(self,
                 sortino_weight: float = 5.0,
                 pnl_weight: float = 0.5,
                 drawdown_weight: float = 10.0,
                 volatility_weight: float = 2.0,
                 transaction_weight: float = 0.1,
                 consistency_weight: float = 2.0,
                 min_history_length: int = 20):
        """
        Initialize reward calculator with configurable weights.

        Args:
            sortino_weight: Weight for Sortino ratio component
            pnl_weight: Weight for raw P&L component
            drawdown_weight: Weight for drawdown penalty
            volatility_weight: Weight for volatility penalty
            transaction_weight: Weight for transaction cost penalty
            consistency_weight: Weight for consistency bonus
            min_history_length: Minimum returns history for Sortino calculation
        """
        self.sortino_weight = sortino_weight
        self.pnl_weight = pnl_weight
        self.drawdown_weight = drawdown_weight
        self.volatility_weight = volatility_weight
        self.transaction_weight = transaction_weight
        self.consistency_weight = consistency_weight
        self.min_history_length = min_history_length

    def calculate_sortino_ratio(self, returns: np.ndarray, target_return: float = 0.0) -> float:
        """
        Calculate Sortino Ratio (downside deviation metric).

        Unlike Sharpe ratio, Sortino only penalizes downside volatility,
        which is more appropriate for trading where upside volatility is desirable.

        Formula: (Mean Return - Target) / Downside Deviation

        Args:
            returns: Array of returns
            target_return: Minimum acceptable return (default 0)

        Returns:
            Sortino ratio value
        """
        if len(returns) < 2:
            return 0.0

        # Calculate mean return
        mean_return = np.mean(returns)

        # Calculate downside deviation (only negative returns)
        downside_returns = returns[returns < target_return]

        if len(downside_returns) == 0:
            # No downside - perfect scenario
            return mean_return / 1e-6 if mean_return > 0 else 0.0

        downside_std = np.std(downside_returns)

        # Avoid division by zero
        if downside_std < 1e-10:
            return mean_return / 1e-6 if mean_return > 0 else 0.0

        sortino = mean_return / downside_std

        return sortino

    def calculate_consistency_score(self, returns: np.ndarray) -> float:
        """
        Calculate consistency score based on positive return ratio.

        Prop firms prefer steady, consistent profits over volatile equity curves.

        Args:
            returns: Array of returns

        Returns:
            Consistency score (0 to 1)
        """
        if len(returns) == 0:
            return 0.0

        # Ratio of positive returns
        positive_ratio = np.sum(returns > 0) / len(returns)

        # Mean return (should be positive for bonus)
        mean_return = np.mean(returns)

        # Only reward consistency if overall returns are positive
        if mean_return > 0:
            return positive_ratio
        else:
            return 0.0

    def calculate_drawdown_penalty(self, current_drawdown: float) -> float:
        """
        Calculate drawdown penalty (quadratic to heavily punish large drawdowns).

        Drawdown is the percentage decline from peak equity.

        Args:
            current_drawdown: Current drawdown percentage (0 to 1)

        Returns:
            Negative penalty value
        """
        # Quadratic penalty - small drawdowns OK, large drawdowns heavily punished
        # For example:
        #   1% drawdown -> -0.0001 penalty
        #   5% drawdown -> -0.0025 penalty
        #  10% drawdown -> -0.01 penalty (very bad)

        penalty = -self.drawdown_weight * (current_drawdown ** 2)

        return penalty

    def calculate_volatility_penalty(self, returns: np.ndarray) -> float:
        """
        Calculate penalty for high volatility.

        Prop firms want smooth equity curves, not erratic performance.

        Args:
            returns: Array of returns

        Returns:
            Negative penalty value
        """
        if len(returns) < 2:
            return 0.0

        volatility = np.std(returns)
        penalty = -self.volatility_weight * volatility

        return penalty

    def calculate_transaction_penalty(self, position_changed: bool) -> float:
        """
        Calculate penalty for trading activity.

        Encourages the agent to be selective and avoid overtrading.

        Args:
            position_changed: Whether position was changed this step

        Returns:
            Negative penalty if trade occurred
        """
        if position_changed:
            return -self.transaction_weight
        return 0.0

    def calculate_reward(self,
                        returns_history: np.ndarray,
                        current_step_pnl_pct: float,
                        current_drawdown_pct: float,
                        position_changed: bool,
                        additional_info: Optional[dict] = None) -> float:
        """
        Calculate the complete reward for current step.

        This is the MAIN reward function that combines all components.

        Args:
            returns_history: Recent returns history (as percentages, e.g., 0.01 = 1%)
            current_step_pnl_pct: P&L for current step (as percentage)
            current_drawdown_pct: Current drawdown from peak (as percentage, 0 to 1)
            position_changed: Whether a trade was executed this step
            additional_info: Optional dict with extra metrics

        Returns:
            Total reward value
        """
        # ====================================================================
        # COMPONENT 1: Sortino Ratio (PRIMARY DRIVER)
        # ====================================================================
        if len(returns_history) >= self.min_history_length:
            sortino_ratio = self.calculate_sortino_ratio(returns_history)
            sortino_component = self.sortino_weight * sortino_ratio
        else:
            # Not enough history yet
            sortino_component = 0.0

        # ====================================================================
        # COMPONENT 2: Raw P&L (SECONDARY)
        # ====================================================================
        # Small weight - we care more about risk-adjusted returns
        pnl_component = self.pnl_weight * current_step_pnl_pct

        # ====================================================================
        # COMPONENT 3: Drawdown Penalty (CRITICAL)
        # ====================================================================
        drawdown_penalty = self.calculate_drawdown_penalty(current_drawdown_pct)

        # ====================================================================
        # COMPONENT 4: Volatility Penalty
        # ====================================================================
        if len(returns_history) >= 2:
            volatility_penalty = self.calculate_volatility_penalty(returns_history)
        else:
            volatility_penalty = 0.0

        # ====================================================================
        # COMPONENT 5: Transaction Cost
        # ====================================================================
        transaction_penalty = self.calculate_transaction_penalty(position_changed)

        # ====================================================================
        # COMPONENT 6: Consistency Bonus
        # ====================================================================
        if len(returns_history) >= self.min_history_length:
            consistency_score = self.calculate_consistency_score(returns_history)
            consistency_bonus = self.consistency_weight * consistency_score
        else:
            consistency_bonus = 0.0

        # ====================================================================
        # COMBINE ALL COMPONENTS
        # ====================================================================
        total_reward = (
            sortino_component +      # Risk-adjusted returns (PRIMARY)
            pnl_component +          # Actual profit (secondary)
            drawdown_penalty +       # Punish drawdowns heavily
            volatility_penalty +     # Prefer stable equity
            transaction_penalty +    # Discourage overtrading
            consistency_bonus        # Reward steady performance
        )

        # Optional: Store component breakdown for debugging
        if additional_info is not None:
            additional_info['reward_breakdown'] = {
                'sortino_component': sortino_component,
                'pnl_component': pnl_component,
                'drawdown_penalty': drawdown_penalty,
                'volatility_penalty': volatility_penalty,
                'transaction_penalty': transaction_penalty,
                'consistency_bonus': consistency_bonus,
                'total_reward': total_reward
            }

        return total_reward

    def get_weights_summary(self) -> dict:
        """
        Get summary of all reward component weights.

        Returns:
            Dictionary of weights
        """
        return {
            'sortino_weight': self.sortino_weight,
            'pnl_weight': self.pnl_weight,
            'drawdown_weight': self.drawdown_weight,
            'volatility_weight': self.volatility_weight,
            'transaction_weight': self.transaction_weight,
            'consistency_weight': self.consistency_weight
        }


class RewardHistory:
    """
    Helper class to track reward history and analyze reward function behavior.
    """

    def __init__(self, maxlen: int = 1000):
        """
        Initialize reward history tracker.

        Args:
            maxlen: Maximum history length to store
        """
        self.rewards = deque(maxlen=maxlen)
        self.components = deque(maxlen=maxlen)

    def add(self, reward: float, components: Optional[dict] = None):
        """
        Add reward and components to history.

        Args:
            reward: Total reward value
            components: Optional dict of reward components
        """
        self.rewards.append(reward)
        if components:
            self.components.append(components)

    def get_statistics(self) -> dict:
        """
        Get statistics about reward history.

        Returns:
            Dictionary of reward statistics
        """
        if len(self.rewards) == 0:
            return {}

        rewards_array = np.array(self.rewards)

        return {
            'mean_reward': np.mean(rewards_array),
            'std_reward': np.std(rewards_array),
            'min_reward': np.min(rewards_array),
            'max_reward': np.max(rewards_array),
            'total_episodes': len(self.rewards)
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage demonstrating reward calculation scenarios.
    """

    print("=" * 80)
    print("REWARD CALCULATOR - TEST MODE")
    print("=" * 80)

    # Initialize calculator
    calc = RewardCalculator()

    print("\nReward weights:")
    for key, value in calc.get_weights_summary().items():
        print(f"  {key}: {value}")

    # -------------------------------------------------------------------------
    # SCENARIO 1: Steady positive returns (IDEAL)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 1: Steady positive returns (IDEAL)")
    print("=" * 80)

    returns = np.array([0.001, 0.0015, 0.0008, 0.0012, 0.0009] * 10)  # 1% average
    drawdown = 0.01  # 1% drawdown
    step_pnl = 0.0012
    position_changed = False

    info = {}
    reward = calc.calculate_reward(returns, step_pnl, drawdown, position_changed, info)

    print(f"Returns: {returns[:5]}... (mean: {np.mean(returns):.4f})")
    print(f"Drawdown: {drawdown:.2%}")
    print(f"Step P&L: {step_pnl:.4f}")
    print(f"\nReward breakdown:")
    for key, value in info['reward_breakdown'].items():
        print(f"  {key}: {value:.4f}")

    # -------------------------------------------------------------------------
    # SCENARIO 2: Large drawdown (BAD)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 2: Large drawdown (BAD)")
    print("=" * 80)

    returns = np.array([-0.02, -0.015, 0.005, -0.01, 0.002] * 10)
    drawdown = 0.08  # 8% drawdown - VERY BAD
    step_pnl = -0.015
    position_changed = True

    info = {}
    reward = calc.calculate_reward(returns, step_pnl, drawdown, position_changed, info)

    print(f"Returns: {returns[:5]}... (mean: {np.mean(returns):.4f})")
    print(f"Drawdown: {drawdown:.2%}")
    print(f"Step P&L: {step_pnl:.4f}")
    print(f"\nReward breakdown:")
    for key, value in info['reward_breakdown'].items():
        print(f"  {key}: {value:.4f}")

    # -------------------------------------------------------------------------
    # SCENARIO 3: High volatility (UNDESIRABLE)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 3: High volatility (UNDESIRABLE)")
    print("=" * 80)

    returns = np.array([0.03, -0.025, 0.028, -0.02, 0.015, -0.018] * 10)
    drawdown = 0.04
    step_pnl = 0.025
    position_changed = True

    info = {}
    reward = calc.calculate_reward(returns, step_pnl, drawdown, position_changed, info)

    print(f"Returns: {returns[:5]}... (mean: {np.mean(returns):.4f})")
    print(f"Volatility: {np.std(returns):.4f}")
    print(f"Drawdown: {drawdown:.2%}")
    print(f"\nReward breakdown:")
    for key, value in info['reward_breakdown'].items():
        print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("1. Scenario 1 (steady gains) gets the highest reward")
    print("2. Scenario 2 (large drawdown) gets heavily penalized")
    print("3. Scenario 3 (volatile) gets penalized despite positive mean return")
    print("\nThis reward function will train a RISK-AVERSE agent suitable for prop firms")
    print("=" * 80)
