"""
Proprietary Trading Firm Safety Layer
======================================

This module implements hard risk guardrails that override AI agent decisions
to ensure compliance with prop firm risk limits.

CRITICAL: This layer operates OUTSIDE the neural network to prevent the agent
from exploiting reward function loopholes. These are non-negotiable rules that
guarantee the account never violates prop firm constraints.

Typical Prop Firm Rules:
- Daily loss limit: 4-5% of account balance
- Maximum drawdown: 8-10% from peak
- Maximum position size limits
- Trading hour restrictions (optional)

Author: DRL Trading System
"""

import numpy as np
from typing import Tuple, Optional
from datetime import datetime, time


class PropFirmSafetyLayer:
    """
    Hard guardrails for prop firm compliance.

    This class acts as a safety net that can override agent actions when
    approaching risk limits. Think of it as a "circuit breaker" that prevents
    catastrophic losses.
    """

    def __init__(self,
                 daily_loss_limit: float = 0.04,        # 4% daily loss
                 max_drawdown_limit: float = 0.10,      # 10% max drawdown
                 daily_loss_buffer: float = 0.005,      # Stop at 3.5% (before hitting 4%)
                 drawdown_buffer: float = 0.02,         # Stop at 8% (before hitting 10%)
                 max_position_size: float = 1.0,        # Maximum position size (lots)
                 enable_logging: bool = True):
        """
        Initialize safety layer with prop firm limits.

        Args:
            daily_loss_limit: Maximum allowed daily loss (as decimal, e.g., 0.04 = 4%)
            max_drawdown_limit: Maximum allowed drawdown from peak
            daily_loss_buffer: Safety buffer for daily loss (stops before limit)
            drawdown_buffer: Safety buffer for drawdown (stops before limit)
            max_position_size: Maximum position size allowed
            enable_logging: Whether to log safety interventions
        """
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_loss_buffer = daily_loss_buffer
        self.drawdown_buffer = drawdown_buffer
        self.max_position_size = max_position_size
        self.enable_logging = enable_logging

        # Tracking variables (initialized by reset())
        self.daily_start_balance = None
        self.peak_balance = None
        self.intervention_count = 0
        self.last_intervention_reason = None

    def reset(self, initial_balance: float):
        """
        Reset safety layer at start of episode/day.

        Args:
            initial_balance: Starting account balance
        """
        self.daily_start_balance = initial_balance
        self.peak_balance = initial_balance
        self.intervention_count = 0
        self.last_intervention_reason = None

    def update_peak_balance(self, current_balance: float):
        """
        Update peak balance for drawdown calculation.

        Args:
            current_balance: Current account balance
        """
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

    def calculate_daily_loss_pct(self, current_balance: float) -> float:
        """
        Calculate current daily loss percentage.

        Args:
            current_balance: Current account balance

        Returns:
            Daily loss percentage (positive number, e.g., 0.03 = 3% loss)
        """
        if self.daily_start_balance is None:
            return 0.0

        daily_loss = self.daily_start_balance - current_balance
        daily_loss_pct = daily_loss / self.daily_start_balance

        return max(0.0, daily_loss_pct)  # Only return positive (losses)

    def calculate_drawdown_pct(self, current_balance: float) -> float:
        """
        Calculate current drawdown from peak.

        Args:
            current_balance: Current account balance

        Returns:
            Drawdown percentage (positive number, e.g., 0.05 = 5% drawdown)
        """
        if self.peak_balance is None:
            return 0.0

        drawdown = self.peak_balance - current_balance
        drawdown_pct = drawdown / self.peak_balance

        return max(0.0, drawdown_pct)

    def check_and_override_action(self,
                                  action: np.ndarray,
                                  current_balance: float,
                                  current_position: float,
                                  unrealized_pnl: float = 0.0) -> Tuple[np.ndarray, bool]:
        """
        Check risk limits and override action if necessary.

        This is the MAIN method called by the trading environment BEFORE
        executing an action from the agent.

        Args:
            action: Agent's proposed action [position_direction, risk_pct]
            current_balance: Current account balance
            current_position: Current position size
            unrealized_pnl: Unrealized P&L on open positions

        Returns:
            Tuple of (modified_action, was_overridden)
        """
        original_action = action.copy()
        was_overridden = False
        intervention_reason = None

        # Update peak balance
        self.update_peak_balance(current_balance + unrealized_pnl)

        # Calculate current risk metrics
        daily_loss_pct = self.calculate_daily_loss_pct(current_balance)
        drawdown_pct = self.calculate_drawdown_pct(current_balance)

        # ====================================================================
        # RULE 1: DAILY LOSS LIMIT (HARD STOP)
        # ====================================================================
        hard_daily_stop = self.daily_loss_limit - self.daily_loss_buffer

        if daily_loss_pct >= hard_daily_stop:
            # CRITICAL: Close all positions and stop trading
            action = np.array([0.0, 0.0], dtype=np.float32)
            was_overridden = True
            intervention_reason = f"DAILY LOSS LIMIT: {daily_loss_pct:.2%} >= {hard_daily_stop:.2%}"

            if self.enable_logging:
                print(f"\n{'!'*80}")
                print(f"[SAFETY LAYER] {intervention_reason}")
                print(f"[SAFETY LAYER] FORCING POSITION CLOSE - NO NEW TRADES ALLOWED")
                print(f"{'!'*80}\n")

        # ====================================================================
        # RULE 2: MAX DRAWDOWN LIMIT (HARD STOP)
        # ====================================================================
        elif drawdown_pct >= (self.max_drawdown_limit - self.drawdown_buffer):
            # CRITICAL: Close all positions and stop trading
            action = np.array([0.0, 0.0], dtype=np.float32)
            was_overridden = True
            intervention_reason = f"MAX DRAWDOWN: {drawdown_pct:.2%} >= {self.max_drawdown_limit - self.drawdown_buffer:.2%}"

            if self.enable_logging:
                print(f"\n{'!'*80}")
                print(f"[SAFETY LAYER] {intervention_reason}")
                print(f"[SAFETY LAYER] FORCING POSITION CLOSE - DRAWDOWN LIMIT REACHED")
                print(f"{'!'*80}\n")

        # ====================================================================
        # RULE 3: RISK REDUCTION ZONE (approaching limits)
        # ====================================================================
        elif daily_loss_pct >= (hard_daily_stop - 0.01):  # Within 1% of hard stop
            # Reduce risk to 30% of normal
            action[1] = min(action[1], 0.3)
            was_overridden = True
            intervention_reason = f"RISK REDUCTION: Daily loss at {daily_loss_pct:.2%}"

            if self.enable_logging:
                print(f"[SAFETY LAYER] {intervention_reason} - Reducing risk to 30%")

        elif drawdown_pct >= (self.max_drawdown_limit - self.drawdown_buffer - 0.02):
            # Within 2% of drawdown limit
            action[1] = min(action[1], 0.5)
            was_overridden = True
            intervention_reason = f"RISK REDUCTION: Drawdown at {drawdown_pct:.2%}"

            if self.enable_logging:
                print(f"[SAFETY LAYER] {intervention_reason} - Reducing risk to 50%")

        # ====================================================================
        # RULE 4: POSITION SIZE LIMITS
        # ====================================================================
        # Ensure position size doesn't exceed maximum
        if np.abs(action[0]) * self.max_position_size > self.max_position_size:
            action[0] = np.sign(action[0]) * 1.0  # Clip to max
            was_overridden = True
            intervention_reason = "POSITION SIZE: Clipped to maximum"

        # ====================================================================
        # RULE 5: ACTION VALIDITY CHECKS
        # ====================================================================
        # Ensure action is within valid bounds
        action[0] = np.clip(action[0], -1.0, 1.0)  # Position direction
        action[1] = np.clip(action[1], 0.0, 1.0)   # Risk percentage

        # Track interventions
        if was_overridden:
            self.intervention_count += 1
            self.last_intervention_reason = intervention_reason

        return action, was_overridden

    def is_trading_allowed(self, current_balance: float) -> bool:
        """
        Check if trading is allowed based on current state.

        Returns:
            True if trading is allowed, False if suspended
        """
        daily_loss_pct = self.calculate_daily_loss_pct(current_balance)
        drawdown_pct = self.calculate_drawdown_pct(current_balance)

        # Suspend trading if approaching limits
        hard_daily_stop = self.daily_loss_limit - self.daily_loss_buffer
        hard_drawdown_stop = self.max_drawdown_limit - self.drawdown_buffer

        if daily_loss_pct >= hard_daily_stop:
            return False

        if drawdown_pct >= hard_drawdown_stop:
            return False

        return True

    def get_risk_status(self, current_balance: float) -> dict:
        """
        Get current risk status metrics.

        Args:
            current_balance: Current account balance

        Returns:
            Dictionary with risk metrics
        """
        daily_loss_pct = self.calculate_daily_loss_pct(current_balance)
        drawdown_pct = self.calculate_drawdown_pct(current_balance)

        # Calculate how close we are to limits (0 = safe, 1 = at limit)
        daily_loss_proximity = daily_loss_pct / self.daily_loss_limit
        drawdown_proximity = drawdown_pct / self.max_drawdown_limit

        return {
            'daily_loss_pct': daily_loss_pct,
            'daily_loss_limit': self.daily_loss_limit,
            'daily_loss_proximity': daily_loss_proximity,
            'drawdown_pct': drawdown_pct,
            'drawdown_limit': self.max_drawdown_limit,
            'drawdown_proximity': drawdown_proximity,
            'trading_allowed': self.is_trading_allowed(current_balance),
            'intervention_count': self.intervention_count,
            'last_intervention': self.last_intervention_reason
        }

    def __repr__(self) -> str:
        """String representation of safety layer configuration."""
        return (f"PropFirmSafetyLayer("
                f"daily_loss_limit={self.daily_loss_limit:.1%}, "
                f"max_drawdown={self.max_drawdown_limit:.1%}, "
                f"interventions={self.intervention_count})")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example demonstrating safety layer in action.
    """

    print("=" * 80)
    print("PROP FIRM SAFETY LAYER - TEST MODE")
    print("=" * 80)

    # Initialize safety layer
    safety = PropFirmSafetyLayer(
        daily_loss_limit=0.04,
        max_drawdown_limit=0.10,
        enable_logging=True
    )

    initial_balance = 100000.0
    safety.reset(initial_balance)

    print(f"\nInitial balance: ${initial_balance:,.2f}")
    print(f"Daily loss limit: {safety.daily_loss_limit:.1%}")
    print(f"Max drawdown limit: {safety.max_drawdown_limit:.1%}")

    # -------------------------------------------------------------------------
    # SCENARIO 1: Normal trading (no intervention)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 1: Normal trading (small loss)")
    print("=" * 80)

    current_balance = 99000.0  # $1000 loss (1%)
    action = np.array([0.5, 0.8])  # Want to go long with 80% risk

    print(f"Current balance: ${current_balance:,.2f} ({(current_balance/initial_balance - 1):.2%})")
    print(f"Agent wants action: {action}")

    modified_action, overridden = safety.check_and_override_action(
        action, current_balance, current_position=0.0
    )

    print(f"Modified action: {modified_action}")
    print(f"Was overridden: {overridden}")

    # -------------------------------------------------------------------------
    # SCENARIO 2: Approaching daily loss limit (risk reduction)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 2: Approaching daily loss limit (3% loss)")
    print("=" * 80)

    current_balance = 97000.0  # $3000 loss (3%)
    action = np.array([0.5, 0.8])

    print(f"Current balance: ${current_balance:,.2f} ({(current_balance/initial_balance - 1):.2%})")
    print(f"Agent wants action: {action}")

    modified_action, overridden = safety.check_and_override_action(
        action, current_balance, current_position=0.0
    )

    print(f"Modified action: {modified_action}")
    print(f"Was overridden: {overridden}")

    # -------------------------------------------------------------------------
    # SCENARIO 3: Hit daily loss limit (HARD STOP)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 3: Daily loss limit breached (3.6% loss)")
    print("=" * 80)

    current_balance = 96400.0  # $3600 loss (3.6% - triggers hard stop at 3.5%)
    action = np.array([0.5, 0.8])

    print(f"Current balance: ${current_balance:,.2f} ({(current_balance/initial_balance - 1):.2%})")
    print(f"Agent wants action: {action}")

    modified_action, overridden = safety.check_and_override_action(
        action, current_balance, current_position=0.5
    )

    print(f"Modified action: {modified_action}")
    print(f"Was overridden: {overridden}")
    print(f"Trading allowed: {safety.is_trading_allowed(current_balance)}")

    # -------------------------------------------------------------------------
    # Risk Status Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RISK STATUS REPORT")
    print("=" * 80)

    status = safety.get_risk_status(current_balance)
    for key, value in status.items():
        if isinstance(value, float):
            if 'pct' in key or 'limit' in key or 'proximity' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print("KEY TAKEAWAY:")
    print("=" * 80)
    print("The safety layer successfully prevents the agent from:")
    print("1. Exceeding daily loss limits")
    print("2. Violating maximum drawdown rules")
    print("3. Taking excessive risk when approaching limits")
    print("\nThis ensures COMPLIANCE with prop firm rules regardless of agent behavior.")
    print("=" * 80)
