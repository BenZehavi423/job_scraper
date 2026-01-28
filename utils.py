import os
import pandas as pd
import datetime

def process_jobs_data(csv_file_path):
    """
    Reads the CSV file and prepares the data for the HTML view.
    """
    if not os.path.exists(csv_file_path):
        return (
            "<div class='text-center p-5 text-muted'>No data yet. Waiting for scheduled scrape...</div>",
            0,
            "Never"
        )

    try:
        df = pd.read_csv(csv_file_path)

        # Determine last update time
        try:
            mod_time = os.path.getmtime(csv_file_path)
            last_update = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        except:
            last_update = "Unknown"

        # Make links clickable
        if 'Link' in df.columns:
            df['Link'] = df['Link'].apply(
                lambda x: f'<a href="{x}" target="_blank" class="job-link">View Job ↗</a>'
            )

        total_jobs = len(df)

        # Convert to HTML table
        table_html = df.to_html(classes='dataframe', index=False, escape=False, border=0)

        return table_html, total_jobs, last_update

    except Exception as e:
        print(f"Error processing CSV: {e}")
        return f"<div class='alert alert-danger'>Error loading data: {e}</div>", 0, "Error"