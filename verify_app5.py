from playwright.sync_api import sync_playwright
import time

def verify_app():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:8501")
        time.sleep(3)  # Wait for Streamlit app to load completely
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1)
        page.screenshot(path="verification_screenshot_ui.png", full_page=True)
        browser.close()

if __name__ == "__main__":
    verify_app()
