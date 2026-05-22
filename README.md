# Feature Selection App

This workspace contains a deployable Streamlit app for feature selection on high-dimensional datasets.

## What is included

- `app.py` — Streamlit application for dataset upload, variance filtering, correlation filtering, and model-based feature selection.
- `feature_selection.py` — reusable feature selection helper functions.
- `requirements.txt` — dependencies for local execution and deployment.

## Run locally

1. Open a terminal in this folder.
2. Create or activate a Python environment with Python 3.10+.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

5. Open the browser URL shown by Streamlit.

## Deploy

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Connect the GitHub repository and branch.
4. Streamlit will use `requirements.txt` and run `app.py` automatically.

### Other platforms

- Use any platform that supports Python and Streamlit.
- Ensure `requirements.txt` is installed.
- Run `streamlit run app.py` or use a container/managed service.

## Notes

- The app supports both a synthetic example dataset and CSV upload.
- If you upload a CSV, select the target column before running selection.
- The final selected feature dataset can be downloaded as CSV.
