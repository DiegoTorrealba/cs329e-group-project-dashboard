# CS329E Group Project Dashboard

## NOTE

- You must clone the repository and run `streamlit run dashboard.py` to launch a local instance in your browser.
- Because of the size of the data, it may take \~3-5 minutes to fully run the script.
- To instead view a markdown, non-iteractive version of the dashboard, click [here](DASHBOARDMARKDOWN.md).

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