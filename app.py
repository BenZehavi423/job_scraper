from flask import Flask, render_template, send_file
from apscheduler.schedulers.background import BackgroundScheduler
import datetime
import os
from scraper import scrape_linkedin
from utils import process_jobs_data

app = Flask(__name__)

# Configuration
CSV_FILE = "jobs_data.csv"
NEXT_RUN_TIME = datetime.datetime.now()


def scheduled_job():
    """
    Scheduled task wrapper.
    Executes the scraper and updates the global next run time.
    """
    global NEXT_RUN_TIME
    print(f" Starting scheduled scrape at {datetime.datetime.now()}")
    try:
        scrape_linkedin()
    except Exception as e:
        print(f"Error during scheduled scrape: {e}")

    # Update next run time
    NEXT_RUN_TIME = datetime.datetime.now() + datetime.timedelta(hours=12)


# Scheduler Setup
scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_job, 'interval', hours=12, next_run_time=datetime.datetime.now())
scheduler.start()


# Routes

@app.route('/')
def index():
    """
    Main Dashboard Route.
    Fetches processed data from utils and renders the HTML template.
    """
    global NEXT_RUN_TIME

    # Use utility function to get data (Logic separated from View)
    table_html, total_jobs, last_update = process_jobs_data(CSV_FILE)

    return render_template(
        'index.html',  # Flask looks for this in the 'templates' folder
        table_html=table_html,
        total_jobs=total_jobs,
        last_update=last_update,
        next_run_iso=NEXT_RUN_TIME.isoformat()
    )


@app.route('/download')
def download():
    """Download the raw CSV file."""
    if os.path.exists(CSV_FILE):
        return send_file(CSV_FILE, as_attachment=True)
    return "File not generated yet.", 404


if __name__ == '__main__':
    print("Server starting on http://127.0.0.1:5000")
    app.run(debug=True, port=5000, use_reloader=False)