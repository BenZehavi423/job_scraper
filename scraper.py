import time
import pandas as pd
import random
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import os

# Configuration
OUTPUT_FILE = "jobs_data.csv"


def load_existing_links():
    """
    Loads existing job links to avoid duplicates.
    """
    if not os.path.exists(OUTPUT_FILE):
        return set()

    try:
        df = pd.read_csv(OUTPUT_FILE)
        if 'Link' in df.columns:
            return set(df['Link'].str.strip())
    except:
        pass
    return set()

def init_driver():
    """
    Initializes the Chrome driver with options to avoid bot detection.
    """
    options = Options()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver


def get_hidden_text_via_js(driver):
    """
    Extracts all text from the DOM using JavaScript.
    This bypasses LinkedIn's "Lazy Loading" and CSS hiding mechanisms.
    """
    try:
        # Get innerText of the body
        text = driver.execute_script("return document.body.innerText;")

        # If body text is too short, try the main container specifically
        if len(text) < 100:
            text = driver.execute_script("""
                var main = document.querySelector('main');
                return main ? main.innerText : document.body.innerText;
             """)
        return text
    except Exception as e:
        print(f"JS Extraction Error: {e}")
        return ""


def extract_rules_based(full_page_text):
    """
    Analyzes the raw text to extract:
    1. Minimum Degree Required (Logic: B.Sc -> M.Sc -> PhD)
    2. Years of Experience
    """
    if not full_page_text:
        return {"degree": "Not Specified", "years_experience": 0}

    text_lower = full_page_text.lower()

    # Extract Years of Experience
    years_pattern = re.search(r'(\d+)(?:\+|\s*-\s*\d+)?\s*(?:years?|yrs?)', text_lower)
    years = 0
    if years_pattern:
        try:
            val = int(years_pattern.group(1))
            # sanity check: experience is usually between 1 and 15
            if 0 < val < 15:
                years = val
        except:
            pass

    # Extract Minimum Degree
    degree = "Not Specified"


    if re.search(r'\bb\.?sc\b|\bbachelor|\bfirst degree\b|\bcomputer science degree\b', text_lower):
        degree = "B.Sc"


    elif re.search(r'\bm\.?sc\b|\bmsc\b|\bmaster|\badvanced degree\b', text_lower):
        degree = "M.Sc"


    elif re.search(r'\bph\.?d\b|\bdoctorate\b', text_lower):
        degree = "PhD"

    return {"degree": degree, "years_experience": years}


def is_relevant_job(title):
    """
    Filters jobs based on the title to ensure relevance.
    Returns True only if the job is a core Data Science role.
    """
    if not title: return False

    title_lower = title.lower()

    # Core keywords
    must_have_keywords = [
        "data scientist",
        "data science",
        "machine learning",
        "deep learning",
        "computer vision",
        "nlp",
        "algorithm",
        "ai engineer",
        "ai researcher",
        "artificial intelligence"
    ]

    # Check for inclusions
    for good_word in must_have_keywords:
        if good_word in title_lower:
            return True

    print(f"DEBUG: Filtered out job '{title}' (Not a core DS role)")
    return False

def get_text_safely(driver, selectors):
    """Helper to try multiple CSS selectors."""
    for selector in selectors:
        try:
            element = driver.find_element(By.CSS_SELECTOR, selector)
            if element.text.strip():
                return element.text.strip()
        except:
            continue
    return None

def scrape_linkedin():
    """
    Main execution function.
    Iterates over LinkedIn search results, extracts data, and saves to CSV.
    """
    driver = init_driver()
    jobs_data = []

    # Check existing links to avoid waist
    existing_links = load_existing_links()
    print(f"Loaded {len(existing_links)} existing jobs from history.")

    try:
        print("Accessing LinkedIn Search...")
        # Search URL for Data Scientist jobs in Israel
        url = "https://www.linkedin.com/jobs/search?keywords=Data%20Scientist&location=Israel&position=1&pageNum=0"
        driver.get(url)
        time.sleep(5)

        # Collect Job Links
        job_links = []
        try:
            # Using multiple selectors to identify job cards
            cards = driver.find_elements(By.CSS_SELECTOR, "a.base-card__full-link, a.job-card-list__title")
            for card in cards[:15]:
                # Taking only 15 jobs to avoid bot detection and long run time
                href = card.get_attribute("href")
                if href:
                    clean_link = href.split('?')[0]
                    if clean_link in existing_links:
                        print(f"Skipping existing job: {clean_link}")
                        continue

                    if clean_link not in job_links:
                        job_links.append(clean_link)

        except Exception as e:
            print(f"Error finding links: {e}")

        print(f"Found {len(job_links)} links. extracting hidden text...")

        # Extract Data from Each Job
        for link in job_links:
            print(f"Scraping: {link}")
            driver.get(link)
            # Wait for DOM to load
            time.sleep(random.uniform(3, 5))

            # Extract Meta Data
            title = get_text_safely(driver, [
                "h1.top-card-layout__title",
                "h1.topcard__title",
                ".job-details-jobs-unified-top-card__job-title h1"
            ])
            if not title:
                # Fallback to page title
                title = driver.title.split(" at ")[0] if " at " in driver.title else "Unknown Job"

            # 2. Company Name Extraction (FIXED)
            company = get_text_safely(driver, [
                "a.topcard__org-name-link",  # Classic Selector
                ".top-card-layout__entity-info a",  # Layout Variation 1
                "span.topcard__flavor",  # Layout Variation 2
                ".job-details-jobs-unified-top-card__company-name a"  # Layout Variation 3
            ])

            if not company and " at " in driver.title:
                # Fallback: Extract from page title "Job at Company | Location"
                try:
                    company = driver.title.split(" at ")[1].split(" |")[0].strip()
                except:
                    company = "Unknown Company"

            if not company:
                company = "Unknown Company"

            # 3. Location Extraction
            location = get_text_safely(driver, [
                "span.topcard__flavor--bullet",
                ".top-card-layout__entity-info:nth-child(2)"
            ]) or "Israel"

            # Filter: Skip irrelevant jobs
            if not is_relevant_job(title):
                print(f"Skipping irrelevant job: {title}")
                continue

            # Extract Full Text via JS
            body_text = get_hidden_text_via_js(driver)

            print(f"DEBUG: Scanned {len(body_text)} chars via JS.")

            # Analyze Text
            insights = extract_rules_based(body_text)

            job_record = {
                "Job Title": title,
                "Company Name": company,
                "Location": "Israel",
                "Degree Required": insights["degree"],
                "Years of Experience": insights["years_experience"],
                "Link": link
            }
            print(f"   -> Extracted: {insights['degree']}, {insights['years_experience']} years")
            jobs_data.append(job_record)

    except Exception as e:
        print(f"Global Error: {e}")

    finally:
        driver.quit()

    # Save to CSV
    if jobs_data:
        df = pd.DataFrame(jobs_data)
        try:
            # Append to existing file if it exists
            existing = pd.read_csv(OUTPUT_FILE)
            combined = pd.concat([existing, df]).drop_duplicates(subset=['Link'], keep='last')
            combined.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        except:
            # Create new file
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"Successfully updated {OUTPUT_FILE}")


if __name__ == "__main__":
    scrape_linkedin()