# LinkedIn Data Scientist Job Scraper

An automated tool designed to scrape, analyze, and monitor Data Scientist job openings in Israel from LinkedIn.
Developed as part of a student position recruitment task.

## Features
* **Automated Scraping:** Collects jobs every 12 hours using a background scheduler.
* **Intelligent Extraction:** Uses Selenium with JavaScript injection to bypass lazy-loading and regex-based logic to extract "Years of Experience" and "Degree Required".
* **Data Persistence:** Saves data to a dynamic CSV file with strict adherence to the requested 6-column format.
* **Web Dashboard:** A Flask-based web interface to view live data, download the CSV, and track the next update time.

## Project Structure
```text
├── app.py              # Main application entry point (Web Server & Scheduler)
├── scraper.py          # Core scraping logic (Selenium & Regex)
├── utils.py            # Data processing utilities
├── templates/
│   └── index.html      # Dashboard UI template
├── requirements.txt    # Python dependencies
└── jobs_data.csv       # Output file (Generated automatically)
```

## Setup & Installation
Follow these steps to set up and run the project locally:

1. Prerequisites:
    * Python 3.8 or higher installed.
   * Google Chrome browser installed (the system uses webdriver-manager to handle the driver automatically).

2. Clone or Download Clone this repository or extract the project files to a local directory.
3. Install Dependencies Install the required Python packages listed in requirements.txt:
```
pip install -r requirements.txt
```
4. Run the Application Start the server and the background scraper:
```
python app.py
```
5. Access the Dashboard Open your web browser and navigate to: http://localhost:5000
   * Note: The scraper will start its first run immediately upon launch. It may take 1-2 minutes for the first data to appear in the dashboard.

## Note on LinkedIn Restrictions
This tool utilizes a "Guest Mode" scraping approach to ensure account safety and avoid bans associated with logged-in bot activity.<br>

* Technical Challenges:LinkedIn aggressively employs anti-scraping measures, such as:
    * Lazy-loading job descriptions (text only loads after user interaction).
    * Hiding full text behind "Show More" buttons.
    * Redirecting guest users to login walls after frequent requests.

* Our Solution: To mitigate this, the system uses a JavaScript Injection Strategy (document.body.innerText) to extract hidden text directly from the DOM, bypassing visual hurdles.

## CSV Output Format
The system generates a file named jobs_data.csv which strictly adheres to the requested format containing only the following 6 columns:
1. Job Title
2. Company Name
3. Location
4. Degree Required
5. Years of Experience
6. Link


-------------------
*Developed by Ben David Zehavi*