"""
Automated Playwright Screenshot Capture Utility for ISM Enterprise DSS v3.0.

Launches (or connects to) the Streamlit Decision Support Dashboard at
http://localhost:8501 with a 1920x1080 Dark Theme viewport and captures
five publication-grade screenshots into ``docs/screenshots/``:
    1. tab1_executive_portfolio.png
    2. tab2_knapsack_optimizer.png
    3. tab3_operational_queue.png
    4. tab3_genai_copilot_script.png
    5. tab5_system_governance.png

Usage:
    .venv\\Scripts\\python.exe scripts/capture_dashboard.py
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from playwright.sync_api import Page, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "docs" / "screenshots"
STREAMLIT_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "streamlit.exe"
APP_FILE = PROJECT_ROOT / "app.py"
DASHBOARD_URL = "http://localhost:8501"


def is_server_running(url: str = DASHBOARD_URL) -> bool:
    """Check whether the Streamlit server is already responding on port 8501."""
    try:
        with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def wait_for_streamlit_idle(page: Page, pause_sec: float = 2.5) -> None:
    """Wait until Streamlit finishes running scripts and rendering Plotly charts."""
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    # Wait for any "Running..." status widget to disappear
    for _ in range(30):
        running_indicators = page.locator('[data-testid="stStatusWidget"]').count()
        if running_indicators == 0:
            break
        time.sleep(0.5)
    time.sleep(pause_sec)


def capture_all_screenshots() -> None:
    """Start Streamlit if needed, navigate through the tabs, and save 5 screenshots."""
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    proc: Optional[subprocess.Popen] = None

    if is_server_running():
        print(f"[SERVER]    Connected to existing Streamlit instance at {DASHBOARD_URL}")
    else:
        print(f"[SERVER]    Starting Streamlit server: {STREAMLIT_EXE} run app.py --server.headless=true")
        proc = subprocess.Popen(
            [
                str(STREAMLIT_EXE),
                "run",
                str(APP_FILE),
                "--server.headless=true",
                "--server.port=8501",
                "--browser.gatherUsageStats=false",
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(40):
            if is_server_running():
                break
            time.sleep(0.5)
        time.sleep(5.0)
        print(f"[SERVER]    Streamlit server ready at {DASHBOARD_URL}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                color_scheme="dark",
                device_scale_factor=1.5,
            )
            page = context.new_page()
            print(f"[BROWSER]   Navigating to {DASHBOARD_URL} (1920x1080 Dark Mode)...")
            page.goto(DASHBOARD_URL, wait_until="domcontentloaded", timeout=45000)
            wait_for_streamlit_idle(page, pause_sec=4.0)

            # -----------------------------------------------------------------
            # 1. Tab 1 — Executive KPI & Portfolio Overview
            # -----------------------------------------------------------------
            out_tab1 = SCREENSHOTS_DIR / "tab1_executive_portfolio.png"
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(1.0)
            page.screenshot(path=str(out_tab1), full_page=True)
            print(f"[CAPTURE 1] Saved {out_tab1.relative_to(PROJECT_ROOT)} ({out_tab1.stat().st_size // 1024} KB)")

            # -----------------------------------------------------------------
            # 2. Tab 2 — What-If, Causal Uplift & Knapsack Optimizer
            # -----------------------------------------------------------------
            tabs = page.locator('button[data-baseweb="tab"]')
            tabs.nth(1).click()
            wait_for_streamlit_idle(page, pause_sec=3.5)
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(1.0)
            out_tab2 = SCREENSHOTS_DIR / "tab2_knapsack_optimizer.png"
            page.screenshot(path=str(out_tab2), full_page=True)
            print(f"[CAPTURE 2] Saved {out_tab2.relative_to(PROJECT_ROOT)} ({out_tab2.stat().st_size // 1024} KB)")

            # -----------------------------------------------------------------
            # 3. Tab 3 (Part A) — Operational Action Queue & Filters
            # -----------------------------------------------------------------
            tabs.nth(2).click()
            wait_for_streamlit_idle(page, pause_sec=3.0)
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(1.0)
            out_tab3_queue = SCREENSHOTS_DIR / "tab3_operational_queue.png"
            page.screenshot(path=str(out_tab3_queue), full_page=False)
            print(
                f"[CAPTURE 3] Saved {out_tab3_queue.relative_to(PROJECT_ROOT)} "
                f"({out_tab3_queue.stat().st_size // 1024} KB)"
            )

            # -----------------------------------------------------------------
            # 4. Tab 3 (Part B) — Customer 360° Inspector & GenAI Copilot Script
            # -----------------------------------------------------------------
            copilot_btn = page.get_by_role("button", name="Generate AI Retention Script & Email", exact=False)
            if copilot_btn.count() > 0:
                copilot_btn.first.scroll_into_view_if_needed()
                time.sleep(0.5)
                copilot_btn.first.click()
                wait_for_streamlit_idle(page, pause_sec=2.5)

            # Scroll specifically to the Customer 360° Inspector & GenAI Copilot section
            inspector_heading = page.get_by_text("Customer 360° Inspector, GenAI Retention Copilot", exact=False)
            if inspector_heading.count() > 0:
                inspector_heading.first.scroll_into_view_if_needed()
                time.sleep(1.0)
            else:
                page.evaluate("window.scrollTo(0, 680)")
                time.sleep(1.0)

            out_tab3_copilot = SCREENSHOTS_DIR / "tab3_genai_copilot_script.png"
            page.screenshot(path=str(out_tab3_copilot), full_page=False)
            print(
                f"[CAPTURE 4] Saved {out_tab3_copilot.relative_to(PROJECT_ROOT)} "
                f"({out_tab3_copilot.stat().st_size // 1024} KB)"
            )

            # -----------------------------------------------------------------
            # 5. Tab 5 — Model Benchmarking & System Governance
            # -----------------------------------------------------------------
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(0.5)
            tabs.nth(4).click()
            wait_for_streamlit_idle(page, pause_sec=3.5)
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(1.0)
            out_tab5 = SCREENSHOTS_DIR / "tab5_system_governance.png"
            page.screenshot(path=str(out_tab5), full_page=True)
            print(f"[CAPTURE 5] Saved {out_tab5.relative_to(PROJECT_ROOT)} ({out_tab5.stat().st_size // 1024} KB)")

            browser.close()
    finally:
        if proc is not None:
            print("[SERVER]    Terminating background Streamlit process...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    capture_all_screenshots()
