#!/usr/bin/env python3
"""Capture screenshots of dashboard HTML files using Playwright."""
import asyncio
from playwright.async_api import async_playwright
import os

async def capture_screenshots():
    html_files = [
        ("docs/dashboard_clean.html", "docs/dashboard_clean.png"),
        ("docs/dashboard_armed.html", "docs/dashboard_armed.png"),
        ("docs/dashboard_threat_detected.html", "docs/dashboard_threat_detected.png"),
        ("docs/dashboard_final.html", "docs/dashboard_final.png"),
    ]
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1400, "height": 900})
        
        for html_file, png_file in html_files:
            if os.path.exists(html_file):
                file_url = f"file://{os.path.abspath(html_file)}"
                await page.goto(file_url)
                await page.wait_for_timeout(500)
                await page.screenshot(path=png_file, full_page=True)
                print(f"[+] Captured {png_file}")
            else:
                print(f"[-] Missing {html_file}")
        
        await browser.close()
    
    print("[+] All screenshots captured!")

asyncio.run(capture_screenshots())
