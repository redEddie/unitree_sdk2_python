"""
Unitree Go2 Mode Manager
=========================

Manages robot mode switching between MCF (Motion Control Framework) mode
and Low-level control mode.

Modes:
- MCF/Sport mode: High-level control, remote controller enabled
- Low-level mode: Direct motor control, remote controller disabled
"""

import time
from typing import Tuple, Optional

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient


class UnitreeModeManager:
    """
    Manages robot mode switching (MCF/Sport mode ↔ Low-level mode).

    This class provides a consistent interface for checking and changing
    robot control modes, matching the behavior of go2_stand_example.py.
    """

    def __init__(self):
        """Initialize mode manager with MotionSwitcherClient and SportClient."""
        # Motion Switcher Client (for mode checking/switching)
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        # Sport Client (for StandDown command)
        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()

    def check_current_mode(self) -> Tuple[int, Optional[dict]]:
        """
        Check current robot mode.

        Returns:
            Tuple of (status_code, result_dict)
            - status_code: 0 if successful
            - result_dict: Contains 'name' key with mode name (e.g., 'normal')
                          None if in low-level mode
        """
        return self.msc.CheckMode()

    def release_mode(self) -> Tuple[int, Optional[dict]]:
        """
        Release current mode.

        Returns:
            Tuple of (status_code, result_dict)
        """
        return self.msc.ReleaseMode()

    def select_mode(self, mode_name: str) -> Tuple[int, Optional[dict]]:
        """
        Select a specific mode.

        Args:
            mode_name: Mode name (e.g., 'mcf', 'normal')

        Returns:
            Tuple of (status_code, result_dict)
        """
        return self.msc.SelectMode(mode_name)

    def disable_sport_mode(self, verbose: bool = True) -> bool:
        """
        Disable Sport mode (MCF) to enable low-level control.

        This method follows the same pattern as go2_stand_example.py:
        1. Check current mode
        2. If MCF is active, call StandDown and ReleaseMode in a loop
        3. Repeat until mode is released

        Args:
            verbose: Whether to print status messages

        Returns:
            True if successful or already in low-level mode
        """
        if verbose:
            print("\n" + "="*70)
            print("Switching to Script Control Mode")
            print("="*70)

        # Check current mode and release if active
        status, result = self.msc.CheckMode()

        # Loop until mode is fully released (same as go2_stand_example.py line 74-79)
        while result and result.get('name'):
            if verbose:
                print(f"Current mode: {result['name']} (releasing...)")

            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        if verbose:
            print("✓ Script control mode enabled")
            print("="*70)

        return True

    def restore_sport_mode(self, verbose: bool = True):
        """
        Restore Sport mode (MCF).

        Note: This is optional and usually not needed. You can manually
        enable Sport mode via remote control if needed.

        Args:
            verbose: Whether to print status messages
        """
        if verbose:
            print("\n" + "="*70)
            print("Note: Sport mode restoration skipped")
            print("You can manually enable Sport mode via remote control if needed")
            print("="*70)

    def toggle_mode(self, verbose: bool = True) -> str:
        """
        Toggle between MCF and Low-level mode.

        Args:
            verbose: Whether to print status messages

        Returns:
            New mode status ('mcf' or 'low_level')
        """
        status, result = self.msc.CheckMode()

        if result and result.get('name'):
            # Currently in MCF mode -> switch to low-level
            if verbose:
                print(f"\nCurrent mode: {result['name']} (MCF/Sport mode)")
                print("Switching to Low-level mode...")

            self.disable_sport_mode(verbose=verbose)
            return 'low_level'
        else:
            # Currently in low-level mode -> switch to MCF
            if verbose:
                print("\nCurrent mode: Low-level mode")
                print("Switching to MCF/Sport mode...")

            self.select_mode("mcf")
            time.sleep(0.5)

            # Verify
            status, result = self.msc.CheckMode()
            if result and result.get('name'):
                if verbose:
                    print(f"✓ Switched to MCF mode: {result['name']}")
                return 'mcf'
            else:
                if verbose:
                    print("⚠️  Warning: Failed to switch to MCF mode")
                return 'low_level'

    def get_mode_status_string(self) -> str:
        """
        Get human-readable mode status.

        Returns:
            Status string describing current mode
        """
        status, result = self.msc.CheckMode()

        if result and result.get('name'):
            return f"MCF/Sport mode ({result['name']}) - Remote control ENABLED, Script control DISABLED"
        else:
            return "Low-level mode - Script control ENABLED, Remote control DISABLED"
