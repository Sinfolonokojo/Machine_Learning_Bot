"""
Fear & Relief Reward Function for DRL Trading
==============================================

This module implements an advanced composite reward function based on behavioral finance
principles. The reward incorporates:

1. **R_skill**: Risk-adjusted returns (log return normalized by volatility)
2. **Fear**: Exponential penalty for approaching daily loss limits ("cliff effect")
3. **Relief**: Reward for drawdown recovery (moving away from the danger zone)
4. **Volatility**: Semi-variance penalty (only penalize negative returns)
5. **Utility**: Logarithmic clamp to prevent lucky spikes from distorting training

Mathematical Formula:
    R_t = Utility(R_skill - Fear - Relief - Volatility)

Where:
    R_skill = log_return / rolling_std_dev
    Fear = 2.0 * (current_drawdown / max_daily_loss) ^ 4
    Relief = 1.5 * (current_drawdown - prev_drawdown)  [sign flipped for reward]
    Volatility = abs(log_return) if log_return < 0 else 0
    Utility(x) = sign(x) * log(1 + abs(x))

Author: Advanced DRL Trading System
"""

import numpy as np
from typing import Optional, Tuple
from collections import deque


class FearReliefRewardCalculator:
    """
    Advanced reward calculator implementing Fear & Relief behavioral economics.

    This reward function creates psychological incentives:
    - Fear: Exponentially punish approaching risk limits (creates "cliff")
    - Relief: Reward drawdown recovery (encourages careful loss recovery)
    - Volatility: Only punish downside volatility (Sortino-like)
    - Utility: Logarithmic smoothing prevents exploitation of lucky wins
    """

    def __init__(self,
                 max_daily_loss_pct: float = 0.04,  # 4% daily loss limit
                 rolling_window: int = 20,           # Window for volatility calculation
                 fear_coefficient: float = 2.0,      # Fear penalty multiplier
                 relief_coefficient: float = 1.5,    # Relief reward multiplier
                 min_std: float = 1e-6):            # Minimum std to avoid division by zero
        """
        Initialize Fear & Relief reward calculator.

        Args:
            max_daily_loss_pct: Maximum allowed daily loss (prop firm limit)
            rolling_window: Window size for rolling std calculation
            fear_coefficient: Multiplier for Fear penalty (default 2.0)
            relief_coefficient: Multiplier for Relief reward (default 1.5)
            min_std: Minimum standard deviation to prevent division errors
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.rolling_window = rolling_window
        self.fear_coefficient = fear_coefficient
        self.relief_coefficient = relief_coefficient
        self.min_std = min_std

        # Track returns history for rolling std
        self.returns_history = deque(maxlen=rolling_window)

        # Track previous drawdown for Relief calculation
        self.prev_drawdown_pct = 0.0

    def reset(self):
        """Reset calculator state at episode start."""
        self.returns_history.clear()
        self.prev_drawdown_pct = 0.0

    def calculate_reward(self,
                        log_return: float,
                        current_drawdown_pct: float,
                        additional_info: Optional[dict] = None) -> float:
        """
        Calculate Fear & Relief composite reward.

        Mathematical Specification:
        $$R_t = \text{Utility}(R_{\text{skill}} - \text{Fear} - \text{Relief} - \text{Volatility})$$

        Args:
            log_return: Log return of current step (ln(price_t / price_{t-1}))
            current_drawdown_pct: Current drawdown percentage (0 to 1)
            additional_info: Optional dict to store component breakdown

        Returns:
            Total reward value
        """
        # Add current return to history
        self.returns_history.append(log_return)

        # ====================================================================
        # COMPONENT 1: R_skill (Risk-Adjusted Return)
        # ====================================================================
        # Formula: log_return / rolling_std_dev

        if len(self.returns_history) >= 2:
            rolling_std_dev = np.std(self.returns_history)
            rolling_std_dev = max(rolling_std_dev, self.min_std)  # Prevent division by zero
        else:
            rolling_std_dev = self.min_std

        r_skill = log_return / rolling_std_dev

        # ====================================================================
        # COMPONENT 2: Fear (Static Drawdown Penalty)
        # ====================================================================
        # Formula: 2.0 * (current_drawdown / max_daily_loss) ^ 4
        # Creates exponential "cliff" effect as limit approaches

        drawdown_ratio = current_drawdown_pct / self.max_daily_loss_pct
        drawdown_ratio = np.clip(drawdown_ratio, 0.0, 1.5)  # Cap at 150% for numerical stability

        fear = self.fear_coefficient * (drawdown_ratio ** 4)

        # ====================================================================
        # COMPONENT 3: Relief (Dynamic Recovery Reward)
        # ====================================================================
        # Formula: 1.5 * (current_drawdown - prev_drawdown)
        # When drawdown decreases (recovery), this becomes negative,
        # so we SUBTRACT it to create a positive reward

        drawdown_change = current_drawdown_pct - self.prev_drawdown_pct
        relief = self.relief_coefficient * drawdown_change

        # Update previous drawdown for next step
        self.prev_drawdown_pct = current_drawdown_pct

        # ====================================================================
        # COMPONENT 4: Volatility (Semi-Variance Penalty)
        # ====================================================================
        # Only penalize volatility if return is negative
        # Formula: abs(log_return) if log_return < 0 else 0

        if log_return < 0:
            volatility = abs(log_return)
        else:
            volatility = 0.0

        # ====================================================================
        # COMBINE COMPONENTS
        # ====================================================================
        # Total before utility function
        total_raw = r_skill - fear - relief - volatility

        # ====================================================================
        # COMPONENT 5: Utility Function (Consistency Clamp)
        # ====================================================================
        # Formula: sign(x) * log(1 + abs(x))
        # Prevents lucky spikes from distorting training

        reward = np.sign(total_raw) * np.log(1.0 + np.abs(total_raw))

        # ====================================================================
        # STORE BREAKDOWN (for debugging/analysis)
        # ====================================================================
        if additional_info is not None:
            additional_info['reward_breakdown'] = {
                'r_skill': r_skill,
                'fear': fear,
                'relief': relief,
                'volatility': volatility,
                'total_raw': total_raw,
                'reward_after_utility': reward,
                'rolling_std_dev': rolling_std_dev,
                'drawdown_ratio': drawdown_ratio
            }

        return reward

    def get_rolling_std(self) -> float:
        """
        Get current rolling standard deviation.

        This is needed for the observation space.

        Returns:
            Current rolling std of returns
        """
        if len(self.returns_history) >= 2:
            return max(np.std(self.returns_history), self.min_std)
        else:
            return self.min_std

    def get_config(self) -> dict:
        """Get configuration summary."""
        return {
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'rolling_window': self.rolling_window,
            'fear_coefficient': self.fear_coefficient,
            'relief_coefficient': self.relief_coefficient
        }


class RewardAnalyzer:
    """
    Helper class to analyze reward function behavior.
    """

    def __init__(self):
        self.rewards = []
        self.components = []

    def add(self, reward: float, components: Optional[dict] = None):
        """Add reward and components to history."""
        self.rewards.append(reward)
        if components:
            self.components.append(components)

    def get_statistics(self) -> dict:
        """Get reward statistics."""
        if len(self.rewards) == 0:
            return {}

        rewards_array = np.array(self.rewards)

        stats = {
            'mean_reward': np.mean(rewards_array),
            'std_reward': np.std(rewards_array),
            'min_reward': np.min(rewards_array),
            'max_reward': np.max(rewards_array),
            'total_episodes': len(self.rewards)
        }

        # Component analysis
        if self.components:
            for key in self.components[0].keys():
                values = [c[key] for c in self.components]
                stats[f'mean_{key}'] = np.mean(values)

        return stats


# ============================================================================
# USAGE EXAMPLE & TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Demonstrate Fear & Relief reward function behavior.
    """

    print("=" * 80)
    print("FEAR & RELIEF REWARD FUNCTION - TEST MODE")
    print("=" * 80)

    calc = FearReliefRewardCalculator()

    print("\nConfiguration:")
    for key, value in calc.get_config().items():
        print(f"  {key}: {value}")

    # -------------------------------------------------------------------------
    # SCENARIO 1: Steady positive returns (IDEAL)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 1: Steady Positive Returns (IDEAL)")
    print("=" * 80)

    calc.reset()
    rewards = []

    for i in range(30):
        log_return = 0.001  # 0.1% positive return
        drawdown = max(0.01 - i * 0.0003, 0.0)  # Decreasing drawdown (recovery)

        info = {}
        reward = calc.calculate_reward(log_return, drawdown, info)
        rewards.append(reward)

        if i % 10 == 0:
            print(f"\nStep {i}:")
            print(f"  Log Return: {log_return:.4f}")
            print(f"  Drawdown: {drawdown:.2%}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Components: {info['reward_breakdown']}")

    print(f"\nAverage reward: {np.mean(rewards):.4f}")

    # -------------------------------------------------------------------------
    # SCENARIO 2: Approaching daily loss limit (FEAR)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 2: Approaching Daily Loss Limit (FEAR KICKS IN)")
    print("=" * 80)

    calc.reset()

    test_drawdowns = [0.01, 0.02, 0.03, 0.035, 0.038, 0.039]

    for dd in test_drawdowns:
        log_return = -0.005  # Negative return

        info = {}
        reward = calc.calculate_reward(log_return, dd, info)

        print(f"\nDrawdown: {dd:.1%} ({dd/calc.max_daily_loss_pct:.0%} of limit)")
        print(f"  Fear component: {info['reward_breakdown']['fear']:.4f}")
        print(f"  Total reward: {reward:.4f}")

    # -------------------------------------------------------------------------
    # SCENARIO 3: Recovery from drawdown (RELIEF)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 3: Drawdown Recovery (RELIEF REWARD)")
    print("=" * 80)

    calc.reset()

    # Simulate recovery: drawdown decreases
    drawdowns = [0.03, 0.028, 0.025, 0.022, 0.02, 0.018, 0.015]

    for i, dd in enumerate(drawdowns):
        log_return = 0.002  # Small positive return

        info = {}
        reward = calc.calculate_reward(log_return, dd, info)

        print(f"\nStep {i}: Drawdown {dd:.2%}")
        print(f"  Relief component: {info['reward_breakdown']['relief']:.4f}")
        print(f"  Total reward: {reward:.4f}")

    # -------------------------------------------------------------------------
    # KEY INSIGHTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("KEY INSIGHTS - Why this reward function works:")
    print("=" * 80)
    print("""
1. R_skill: Rewards risk-adjusted returns (not just raw profits)
   - Prevents the agent from taking excessive risk for small gains

2. Fear: Creates exponential "cliff" near daily loss limit
   - At 1% drawdown: Small penalty
   - At 3.5% drawdown (near 4% limit): MASSIVE penalty
   - Forces agent to be extremely cautious near limits

3. Relief: Rewards gradual recovery from drawdowns
   - Encourages patient loss recovery rather than revenge trading
   - Creates positive feedback for risk reduction
   - Mathematical insight: When drawdown decreases, relief < 0,
     so -relief becomes positive reward

4. Volatility: Only penalizes negative returns
   - Sortino-like: doesn't punish upside volatility
   - Aligns with trader psychology

5. Utility Function: Logarithmic smoothing
   - Prevents one lucky trade from dominating training signal
   - Encourages consistency over volatility
   - Makes reward more stationary for RL training

RESULT: Agent learns to:
- Maintain small, consistent profits
- Avoid approaching risk limits at all costs
- Recover losses gradually and safely
- Prefer stable equity curves over volatile wins
    """)
    print("=" * 80)
