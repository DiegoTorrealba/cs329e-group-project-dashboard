# CS329E Group Project Dashboard

## NOTE

- You must clone the repository and run `streamlit run dashboard.py` to launch a local instance in your browser.
- Because of the size of the data, it may take \~3-5 minutes to run the script entirely.
- To instead view a markdown, non-interactive version of the dashboard, click [here](DASHBOARDMARKDOWN.md).

### **Contributors**:

-   Diego Torrealba
- Jenna Nega
-   Alexander Domond

### **Description**:

This repo contains the source code and necessary files to run the [Streamlit](https://streamlit.io/) dashboard for our class final project on [Austin, TX, crime data](https://data.austintexas.gov/Public-Safety/Crime-Reports/fdj4-gpfu/about_data) from 2003 to 2025. This project explores patterns in crime types, locations, frequency over time, and uses machine learning (Random Forest) to classify crime-related outcomes.

### **Important packages used in this project**:

*Data Manipulation*:

- [Pandas](https://pandas.pydata.org/)
- [GeoPandas](https://geopandas.org/en/stable/)
- [Shapely](https://shapely.readthedocs.io/en/stable/)

*Data Visualization*:

- [Vega-Altair](https://altair-viz.github.io/)
- [Plotly](https://plotly.com/)

*Machine learning*:

- [Scikit-learn](https://scikit-learn.org/stable/)
- [Statsmodels](https://www.statsmodels.org/stable/index.html)

### Running on your local machine 
1. Create a Python virtual environment (venv, conda, uv, etc.) after cloning the repo in your desired location. If you are using venv: `python -m venv .venv`.
2. Activate the virtual environment. If you use `venv`: `source .venv/bin/activate` (macOS/Linux) OR `.venv\Scripts\activate` (Windows).
3. Install packages using pip: `pip install -r requirements.txt`.
4. Once packages are installed, to run the app, type this command in the terminal while in the directory of the `austincrimedash`: `streamlit run dashboard.py`. This will launch a local instance of the app in your browser.
5. Kill the terminal using `Ctrl+C` once you finish the application. 
   
