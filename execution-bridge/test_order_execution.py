"""
Test C: Order Execution via Kore Event Streams
================================================
Uploads the execution-bridge + test-signal-sender strategies to ForwardTest,
starts them, monitors running status, and checks order/trade activity.

This test verifies:
  1. Strategy upload to Kore ForwardTest
  2. Event stream communication (publish/subscribe between strategies)
  3. Buy order execution (10 symbols, market orders)
  4. Stop loss monitoring (0.5% below entry)
  5. Sell order execution (exit via stop loss or safety timeout)
  6. Fill/ack/reject event flow

Architecture:
  test_signal_sender  --[!order_signals]--> execution_bridge
  test_signal_sender <--[!order_events]---  execution_bridge
                                                |
                                          order.algo_buy()
                                          order.algo_sell()
                                                |
                                          Kore OMS / Exchange

Usage:
  python test_order_execution.py                  # Full flow: upload + start + monitor
  python test_order_execution.py --upload-only     # Just upload strategies
  python test_order_execution.py --start-only      # Just start (already uploaded)
  python test_order_execution.py --monitor-only    # Just monitor running strategies
  python test_order_execution.py --stop            # Stop all test strategies
"""

import subprocess
import json
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ACCOUNT = "203979"
BRIDGE_CODE = "code.py"
BRIDGE_META = "meta.yaml"
SENDER_CODE = "test-signal-sender/code.py"
SENDER_META = "test-signal-sender/meta.yaml"

TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                "META", "TSLA", "JPM", "V", "MA"]

# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------

class T:
    B = "\033[1m"
    D = "\033[2m"
    G = "\033[92m"
    Y = "\033[93m"
    R = "\033[91m"
    C = "\033[96m"
    M = "\033[95m"
    W = "\033[97m"
    X = "\033[0m"

    @staticmethod
    def header(text):
        print(f"\n{T.B}{T.C}{'=' * 70}")
        print(f"  {text}")
        print(f"{'=' * 70}{T.X}\n")

    @staticmethod
    def step(n, text):
        print(f"  {T.B}{T.W}[Step {n}]{T.X}  {text}")

    @staticmethod
    def ok(text):
        print(f"  {T.G}[OK]{T.X}      {text}")

    @staticmethod
    def fail(text):
        print(f"  {T.R}[FAIL]{T.X}    {text}")

    @staticmethod
    def warn(text):
        print(f"  {T.Y}[!!]{T.X}      {text}")

    @staticmethod
    def info(text):
        print(f"  {T.D}[..]{T.X}      {text}")

    @staticmethod
    def detail(text):
        print(f"            {T.D}{text}{T.X}")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def run(args, timeout=60):
    """Run a shell command, return (stdout, stderr, returncode)."""
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", 1


def run_kti(args, timeout=60):
    """Run a kti command, return (stdout, stderr, returncode)."""
    return run(["kti"] + args, timeout)


# ---------------------------------------------------------------------------
# Phase 1: Upload strategies
# ---------------------------------------------------------------------------

def upload_strategies():
    T.header("PHASE 1: Upload Strategies to ForwardTest")

    # Upload execution bridge
    T.step(1, "Uploading execution-bridge strategy...")
    T.info(f"Code: {BRIDGE_CODE}")
    T.info(f"Meta: {BRIDGE_META}")

    out, err, rc = run_kti([
        "strategy", "upload",
        "--code", BRIDGE_CODE,
        "--meta", BRIDGE_META,
    ])

    if rc != 0:
        T.fail(f"Upload failed!")
        for line in (err or out).split("\n"):
            T.detail(line)
        return None, None

    # Parse GUID from output
    bridge_guid = None
    for line in out.split("\n"):
        print(f"            {line}")
        if "GUID:" in line:
            bridge_guid = line.split("GUID:")[1].strip()

    if bridge_guid:
        T.ok(f"execution-bridge uploaded: {bridge_guid}")
    else:
        T.warn("Could not parse GUID from output")
        # Try to find it from strategy list
        bridge_guid = find_strategy_guid("execution_bridge")

    print()

    # Upload test signal sender
    T.step(2, "Uploading test-signal-sender strategy...")
    T.info(f"Code: {SENDER_CODE}")
    T.info(f"Meta: {SENDER_META}")

    out, err, rc = run_kti([
        "strategy", "upload",
        "--code", SENDER_CODE,
        "--meta", SENDER_META,
    ])

    if rc != 0:
        T.fail(f"Upload failed!")
        for line in (err or out).split("\n"):
            T.detail(line)
        return bridge_guid, None

    sender_guid = None
    for line in out.split("\n"):
        print(f"            {line}")
        if "GUID:" in line:
            sender_guid = line.split("GUID:")[1].strip()

    if sender_guid:
        T.ok(f"test-signal-sender uploaded: {sender_guid}")
    else:
        T.warn("Could not parse GUID from output")
        sender_guid = find_strategy_guid("test_signal_sender")

    return bridge_guid, sender_guid


def find_strategy_guid(name):
    """Look up strategy GUID by name from strategy list."""
    out, err, rc = run_kti(["strategy", "list", "--json"])
    if rc != 0:
        return None
    try:
        data = json.loads(out)
        for s in data:
            if s.get("name", "").lower() == name.lower():
                return s.get("guid")
    except (json.JSONDecodeError, KeyError):
        pass
    return None


# ---------------------------------------------------------------------------
# Phase 2: Start strategies
# ---------------------------------------------------------------------------

def start_strategies(bridge_guid, sender_guid):
    T.header("PHASE 2: Start Strategies on ForwardTest")

    if not bridge_guid or not sender_guid:
        T.fail("Missing strategy GUIDs, cannot start")
        T.info(f"Bridge GUID:  {bridge_guid}")
        T.info(f"Sender GUID:  {sender_guid}")
        return False

    T.info(f"Account:      {ACCOUNT}")
    T.info(f"Bridge GUID:  {bridge_guid}")
    T.info(f"Sender GUID:  {sender_guid}")
    T.info(f"Symbols:      {', '.join(TEST_SYMBOLS)}")
    print()

    # Start execution-bridge on all symbols (it qualifies all)
    T.step(1, "Starting execution-bridge on all symbols...")
    T.info("This strategy qualifies ALL symbols so it can receive signals for any ticker")

    out, err, rc = run_kti([
        "strategy", "start", bridge_guid,
        "--account", ACCOUNT,
        "--all-symbols",
    ], timeout=30)

    if rc != 0:
        T.fail("Start failed!")
        for line in (err or out).split("\n"):
            T.detail(line)
        return False

    for line in out.split("\n"):
        print(f"            {line}")
    T.ok("execution-bridge start command sent")
    print()

    # Brief pause to let bridge initialize
    T.info("Waiting 5s for execution-bridge to initialize...")
    time.sleep(5)

    # Start test-signal-sender on all symbols (it qualifies only TEST_SYMBOLS)
    T.step(2, "Starting test-signal-sender on all symbols...")
    T.info("This strategy only qualifies TEST_SYMBOLS via is_symbol_qualified")

    out, err, rc = run_kti([
        "strategy", "start", sender_guid,
        "--account", ACCOUNT,
        "--all-symbols",
    ], timeout=30)

    if rc != 0:
        T.fail("Start failed!")
        for line in (err or out).split("\n"):
            T.detail(line)
        return False

    for line in out.split("\n"):
        print(f"            {line}")
    T.ok("test-signal-sender start command sent")

    return True


# ---------------------------------------------------------------------------
# Phase 3: Monitor
# ---------------------------------------------------------------------------

def monitor_strategies(duration_seconds=180):
    T.header("PHASE 3: Monitor Running Strategies")

    T.info(f"Account:    {ACCOUNT}")
    T.info(f"Duration:   {duration_seconds}s ({duration_seconds // 60}min)")
    T.info(f"Checking every 15s for running status and order activity")
    T.info(f"Press Ctrl+C to stop monitoring early")
    print()

    start_time = time.time()
    poll = 0

    try:
        while time.time() - start_time < duration_seconds:
            poll += 1
            elapsed = int(time.time() - start_time)
            now = datetime.now().strftime("%H:%M:%S")

            print(f"  {T.B}{T.C}[Poll #{poll}  {now}  elapsed={elapsed}s]{T.X}")

            # Check running strategies
            out, err, rc = run_kti([
                "forward", "running",
                "--account", ACCOUNT,
            ])

            if rc == 0 and out:
                for line in out.split("\n"):
                    if line.strip():
                        T.detail(line)
            elif rc != 0:
                T.warn(f"Cannot check running status: {err[:100]}")
                T.info("This may require account access permissions")

            # Check recent orders via order log
            out, err, rc = run_kti([
                "order", "log",
                "--account", ACCOUNT,
                "--date", "today",
            ])

            if rc == 0 and out:
                lines = out.strip().split("\n")
                if len(lines) > 1:  # has data beyond header
                    T.info(f"Order activity ({len(lines) - 1} entries):")
                    for line in lines[:15]:  # show first 15 lines
                        T.detail(line)
                    if len(lines) > 15:
                        T.detail(f"... and {len(lines) - 15} more")
                else:
                    T.info("No order activity yet")
            elif rc != 0:
                # Try alternate order commands
                pass

            print()

            if elapsed + 15 < duration_seconds:
                time.sleep(15)
            else:
                break

    except KeyboardInterrupt:
        print(f"\n  {T.B}Monitoring stopped by user.{T.X}\n")


# ---------------------------------------------------------------------------
# Phase 4: Check results
# ---------------------------------------------------------------------------

def check_results():
    T.header("PHASE 4: Check Trade Results")

    T.step(1, "Checking today's trades on the account...")

    out, err, rc = run_kti([
        "trades", "report",
        "--account", ACCOUNT,
        "--date", "today",
    ])

    if rc == 0 and out:
        for line in out.split("\n"):
            if line.strip():
                print(f"            {line}")
    else:
        T.info("No trades found or cannot access trade report")
        if err:
            T.detail(err[:200])

    print()
    T.step(2, "Checking order log for today...")

    out, err, rc = run_kti([
        "order", "log",
        "--account", ACCOUNT,
        "--date", "today",
        "--json",
    ])

    if rc == 0 and out:
        try:
            orders = json.loads(out)
            if isinstance(orders, list):
                T.info(f"Total orders: {len(orders)}")
                buys = [o for o in orders if "buy" in str(o.get("side", "")).lower()]
                sells = [o for o in orders if "sell" in str(o.get("side", "")).lower()]
                T.info(f"Buy orders:   {len(buys)}")
                T.info(f"Sell orders:  {len(sells)}")

                fills = [o for o in orders if "fill" in str(o.get("state", "")).lower()]
                rejects = [o for o in orders if "reject" in str(o.get("state", "")).lower()]
                T.info(f"Fills:        {len(fills)}")
                T.info(f"Rejects:      {len(rejects)}")
            else:
                T.info("Order data format unexpected")
        except json.JSONDecodeError:
            # Not JSON, show raw
            for line in out.split("\n")[:20]:
                T.detail(line)
    else:
        T.info("Cannot retrieve order log")
        if err:
            T.detail(err[:200])


# ---------------------------------------------------------------------------
# Stop strategies
# ---------------------------------------------------------------------------

def stop_strategies():
    T.header("Stopping Test Strategies")

    # Find our strategy GUIDs
    bridge_guid = find_strategy_guid("execution_bridge")
    sender_guid = find_strategy_guid("test_signal_sender")

    for name, guid in [("execution_bridge", bridge_guid), ("test_signal_sender", sender_guid)]:
        if not guid:
            T.warn(f"{name}: GUID not found, skipping")
            continue

        T.step("", f"Stopping {name} ({guid})...")
        out, err, rc = run_kti([
            "forward", "stop", guid,
            "--account", ACCOUNT,
        ])
        if rc == 0:
            T.ok(f"{name} stopped")
            for line in out.split("\n"):
                if line.strip():
                    T.detail(line)
        else:
            T.warn(f"Could not stop {name}: {(err or out)[:100]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    T.header("TEST C: ORDER EXECUTION VIA KORE EVENT STREAMS")
    T.info(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    T.info(f"Account:  {ACCOUNT} (ForwardTest)")
    T.info(f"Symbols:  {', '.join(TEST_SYMBOLS)}")
    T.info(f"Strategy: execution_bridge (order router)")
    T.info(f"Strategy: test_signal_sender (test driver)")
    print()

    T.info("Architecture:")
    T.detail("test_signal_sender  ---[!order_signals]--->  execution_bridge")
    T.detail("                                                   |")
    T.detail("                                             order.algo_buy()")
    T.detail("                                             order.algo_sell()")
    T.detail("                                                   |")
    T.detail("test_signal_sender  <--[!order_events]----  execution_bridge")
    print()

    T.info("Test plan:")
    T.detail("1. Upload both strategies to ForwardTest")
    T.detail("2. Start execution_bridge on ALL symbols")
    T.detail("3. Start test_signal_sender on 10 test symbols")
    T.detail("4. test_signal_sender sends BUY signals (10s after start)")
    T.detail("5. execution_bridge routes market buy orders")
    T.detail("6. Fill events flow back, stop losses set at 0.5% below entry")
    T.detail("7. Price monitoring: sell if stop hit, or safety exit at 5min")
    T.detail("8. Monitor order flow and trade results")
    print()

    # Parse args
    if "--upload-only" in sys.argv:
        upload_strategies()
        return

    if "--start-only" in sys.argv:
        bridge_guid = find_strategy_guid("execution_bridge")
        sender_guid = find_strategy_guid("test_signal_sender")
        if bridge_guid and sender_guid:
            start_strategies(bridge_guid, sender_guid)
        else:
            T.fail("Strategies not found. Run without --start-only first to upload.")
            T.info(f"Bridge: {bridge_guid}")
            T.info(f"Sender: {sender_guid}")
        return

    if "--monitor-only" in sys.argv:
        monitor_strategies(duration_seconds=300)
        check_results()
        return

    if "--stop" in sys.argv:
        stop_strategies()
        return

    # Full flow
    bridge_guid, sender_guid = upload_strategies()

    if not bridge_guid or not sender_guid:
        T.fail("Upload incomplete. Fix errors above and retry.")
        return

    T.info("Both strategies uploaded successfully!")
    print()

    # Confirm before starting (this will place real orders on forward test)
    print(f"  {T.B}{T.Y}*** ABOUT TO START LIVE FORWARD TEST ***{T.X}")
    print(f"  {T.Y}This will place real orders on account {ACCOUNT} (ForwardTest){T.X}")
    print(f"  {T.Y}10 market buy orders ({TEST_SYMBOLS[0]}, {TEST_SYMBOLS[1]}, ... {TEST_SYMBOLS[-1]}){T.X}")
    print(f"  {T.Y}Followed by sell orders (stop loss or 5min timeout){T.X}")
    print()
    answer = input(f"  {T.B}Proceed? [y/N]: {T.X}").strip().lower()
    if answer != "y":
        print(f"\n  Aborted. Strategies are uploaded but not started.")
        print(f"  To start later:  python test_order_execution.py --start-only")
        print(f"  To monitor:      python test_order_execution.py --monitor-only")
        return

    print()
    started = start_strategies(bridge_guid, sender_guid)

    if not started:
        T.fail("Failed to start strategies. Check errors above.")
        return

    T.ok("Both strategies started!")
    T.info("Orders will begin in ~10 seconds (timer delay)")
    T.info("Stop losses will trigger at 0.5% below entry price")
    T.info("Safety exit after 5 minutes if stops not hit")
    print()

    # Monitor for 6 minutes (5min safety exit + 1min buffer)
    monitor_strategies(duration_seconds=360)

    # Check final results
    check_results()

    T.header("TEST COMPLETE")
    T.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    T.info(f"To stop strategies: python test_order_execution.py --stop")
    T.info(f"To re-monitor:      python test_order_execution.py --monitor-only")
    print()


if __name__ == "__main__":
    main()
