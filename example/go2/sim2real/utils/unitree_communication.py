"""
Unitree Go2 Communication Utilities
====================================

Centralized communication setup and channel management for Unitree SDK2.
"""

import time
from typing import Optional, Callable

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelSubscriber,
    ChannelPublisher
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_, SportModeState_


# =============================================================================
# Channel Factory Initialization
# =============================================================================

def initialize_channel_factory(network_interface: Optional[str] = None, verbose: bool = True) -> None:
    """
    Initialize ChannelFactory with optional network interface.

    Args:
        network_interface: Network interface name (e.g., 'eth0', 'eno1')
                          None or 'default' uses default interface
        verbose: Whether to print initialization message
    """
    if network_interface is None or network_interface == 'default':
        if verbose:
            print("Initializing communication on default interface")
        ChannelFactoryInitialize(0)
    else:
        if verbose:
            print(f"Initializing communication on network interface: {network_interface}")
        ChannelFactoryInitialize(0, network_interface)


# =============================================================================
# Subscriber Creation
# =============================================================================

def create_lowstate_subscriber(
    handler_callback: Callable[[LowState_], None],
    queue_size: int = 10
) -> ChannelSubscriber:
    """
    Create and initialize LowState subscriber.

    Args:
        handler_callback: Callback function to handle incoming LowState messages
        queue_size: Size of message queue

    Returns:
        Initialized ChannelSubscriber
    """
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(handler_callback, queue_size)
    return subscriber


def create_sportmode_subscriber(
    handler_callback: Callable[[SportModeState_], None],
    queue_size: int = 10
) -> ChannelSubscriber:
    """
    Create and initialize SportModeState subscriber.

    Note: This topic is only available when MCF (Sport mode) is active.

    Args:
        handler_callback: Callback function to handle incoming SportModeState messages
        queue_size: Size of message queue

    Returns:
        Initialized ChannelSubscriber
    """
    subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    subscriber.Init(handler_callback, queue_size)
    return subscriber


# =============================================================================
# Publisher Creation
# =============================================================================

def create_lowcmd_publisher() -> ChannelPublisher:
    """
    Create and initialize LowCmd publisher.

    Returns:
        Initialized ChannelPublisher
    """
    publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    publisher.Init()
    return publisher


# =============================================================================
# Data Waiting Utilities
# =============================================================================

def wait_for_robot_data(
    lowstate_ref: list,
    sportmode_ref: Optional[list] = None,
    timeout: float = 5.0,
    verbose: bool = True
) -> bool:
    """
    Wait for robot data with timeout.

    This function waits for LowState data (required) and optionally
    SportModeState data (only available when MCF is active).

    Args:
        lowstate_ref: List containing latest LowState (modified in-place)
                     Format: [LowState_ or None]
        sportmode_ref: Optional list containing latest SportModeState
                      Format: [SportModeState_ or None]
                      If None, only waits for LowState
        timeout: Timeout in seconds
        verbose: Whether to print status messages

    Returns:
        True if data received successfully

    Raises:
        TimeoutError: If LowState not received within timeout
    """
    if verbose:
        print("\nWaiting for robot data...")
        if sportmode_ref is None:
            print("(Note: rt/sportmodestate only available when MCF is active)")

    start_time = time.time()

    # Wait for LowState only (always available)
    while lowstate_ref[0] is None:
        if time.time() - start_time > timeout:
            raise TimeoutError("Failed to receive LowState within timeout")
        time.sleep(0.01)

    if verbose:
        print("✓ Data received from robot!")
        print(f"  LowState: ✓")

    # Check SportModeState if requested
    if sportmode_ref is not None:
        if sportmode_ref[0] is not None:
            if verbose:
                print(f"  SportModeState: ✓ (MCF active)")
        else:
            if verbose:
                print(f"  SportModeState: ✗ (MCF off)")

    return True


class RobotDataWaiter:
    """
    Helper class for waiting for robot data in a more object-oriented way.

    This class can be used as an alternative to the functional API.
    """

    def __init__(self):
        self.latest_lowstate: Optional[LowState_] = None
        self.latest_sportmode: Optional[SportModeState_] = None

    def lowstate_handler(self, msg: LowState_):
        """Handler for LowState messages."""
        self.latest_lowstate = msg

    def sportmode_handler(self, msg: SportModeState_):
        """Handler for SportModeState messages."""
        self.latest_sportmode = msg

    def wait_for_data(
        self,
        require_sportmode: bool = False,
        timeout: float = 5.0,
        verbose: bool = True
    ) -> bool:
        """
        Wait for robot data.

        Args:
            require_sportmode: Whether to require SportModeState
            timeout: Timeout in seconds
            verbose: Whether to print status messages

        Returns:
            True if data received successfully

        Raises:
            TimeoutError: If required data not received within timeout
        """
        if verbose:
            print("\nWaiting for robot data...")

        start_time = time.time()

        # Wait for LowState
        while self.latest_lowstate is None:
            if time.time() - start_time > timeout:
                raise TimeoutError("Failed to receive LowState within timeout")
            time.sleep(0.01)

        # Wait for SportModeState if required
        if require_sportmode:
            while self.latest_sportmode is None:
                if time.time() - start_time > timeout:
                    raise TimeoutError("Failed to receive SportModeState within timeout")
                time.sleep(0.01)

        if verbose:
            print("✓ Data received from robot!")
            print(f"  LowState: ✓")
            if self.latest_sportmode is not None:
                print(f"  SportModeState: ✓")
            elif not require_sportmode:
                print(f"  SportModeState: ✗ (not required)")

        return True

    def reset(self):
        """Reset stored data."""
        self.latest_lowstate = None
        self.latest_sportmode = None
