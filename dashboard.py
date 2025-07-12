import json
import pandas as pd
import altair as alt
import plotly.express as px
import geopandas as gpd
import plotly.graph_objects as go
import streamlit as st
from shapely.geometry import Point
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('austin_crime_v2.csv')

df.loc[2453275, 'Category'] = 'human trafficking'

# fig1

cat_off_df = pd.DataFrame(df[['Category', 'Offense']].value_counts())
cat_off_df.reset_index(drop=False, inplace=True)

fig1 = px.treemap(cat_off_df, 
                 path = ['Category', 'Offense'], 
                 values='count',
                 color = 'count',
                 color_continuous_scale='balance',
                 title='Categories of Offenses by Number of Records from APD data',
                 
)
fig1.update_layout(margin = dict(t=75, l=25, r=25, b=25))
fig1.update_layout(title = {'subtitle' : {'text': 'Theft and family-related crimes on the surface seem to have the most number of records.'}})

# fig2

fig2 = pd.DataFrame(df[['Category', 'Offense','Time of Day']].value_counts())
fig2.reset_index(drop=False, inplace=True)
category_sum = fig2.groupby('Category')['count'].sum().reset_index()
category_sum.columns = ['Category', 'cat_count']
TODorder = ["Morning", "Afternoon", "Evening", "Night"]
fig2['Time of Day'] = pd.Categorical(fig2['Time of Day'], categories=TODorder, ordered=True)
fig2 = fig2.merge(category_sum, on='Category', how='left')

selection = alt.selection_single(
    fields=['Category'],
    empty='all'
)

frequency = alt.Chart(fig2).mark_bar().encode(
    alt.X('Category:N', title="Crime Category"),
    alt.Y('count:Q', title="Frequency"),
    alt.Color('Time of Day:O',scale=alt.Scale
              (domain=["Morning", "Afternoon", "Evening", "Night"]
                ,range=["#F1C40F", "#5DADE2", "#4C78A8", "#2C3E50"])),
    opacity= alt.condition(selection, alt.value(1), alt.value(0.3)),
    tooltip=[alt.Tooltip('cat_count:N', title="Total Count"),
              alt.Tooltip('Offense:N', title="Offense"),
              alt.Tooltip('count:Q', title="Count"),
              alt.Tooltip('Category:N', title="Crime Category")]

).properties(
    title='Austin Crime Frequency Distribution',
    width=500,
).add_selection(
    selection
)

time = alt.Chart(fig2).mark_bar().encode(
    alt.X('Time of Day:O', title="Time of Day", sort=["Morning", "Afternoon", "Evening", "Night"]),
    alt.Y('count:Q', title="Frequency"),
    alt.Color('Time of Day:N',scale=alt.Scale
              (domain=["Morning", "Afternoon", "Evening", "Night"]
                ,range=["#F1C40F", "#5DADE2", "#4C78A8", "#2C3E50"])),
    opacity= alt.condition(selection, alt.value(1), alt.value(0.3)),
    tooltip=[alt.Tooltip('cat_count:N', title="Total Count"),
              alt.Tooltip('Offense:N', title="Offense"),
              alt.Tooltip('count:Q', title="Count"),
              alt.Tooltip('Category:N', title="Crime Category")]
    
).transform_filter(
    selection
).properties(
    title='Time of Day Distribution',
    width=500,
)

fig2 = frequency & time

# fig3

with open('austin_neighborhoods.geojson', 'r') as file:
  austin_neighborhoods = json.load(file)
  
austin_neighborhoods_df = gpd.read_file('austin_neighborhoods.geojson')

df['geometry'] = df.apply(lambda x: Point(x['Longitude'], x['Latitude']), axis = 1)
df3 = df.copy()

df3 = df3[ (df3['Latitude'].isna() == False) | (df3['Longitude'].isna() == False)]

df3_gdf = gpd.GeoDataFrame(df3, geometry='geometry', crs='EPSG:4326')
df3_gdf = gpd.sjoin(df3_gdf, austin_neighborhoods_df, how='inner', predicate='within')

df3_gdf = df3_gdf.groupby(['neighname', 'Category']).size().reset_index(name = 'Count')

neighborhoods = df3_gdf['neighname'].unique()
categories = df3_gdf['Category'].unique()

df3_viz_mi = pd.MultiIndex.from_product([neighborhoods, categories], names = ['neighname', 'Category'])
df3_gdf = df3_gdf.set_index(['neighname', 'Category'])
df3_gdf = df3_gdf.reindex(df3_viz_mi, fill_value=0).reset_index()

fig3 = px.choropleth_map(
  df3_gdf[(df3_gdf['Category'] == 'abuse')],
  geojson=austin_neighborhoods,
  color='Count',
  locations = 'neighname',
  featureidkey='properties.neighname',
  center= {'lat':30.266666, 'lon':-97.733330},
  title= f'Crime Distributions by Austin neighborhoods<br><sup>Distribution of Abuse crimes by Austin neighborhoods</sup>',
  hover_data={'neighname' : True, 'Count':True},
  labels = {'neighname' : 'Neighborhood',
             'Count' : 'Number of<br>Incidents'},
  width = 800,
  height = 500,
  opacity= 0.6
)


fig3.update_layout(
  updatemenus = [
    dict(
      buttons = list([
        dict(
          args = [{
            'z' : [df3_gdf[df3_gdf['Category'] == crime]['Count'].tolist()]
          },
                  {
                    'title.text': f'Crime Distributions by Austin neighborhoods<br><sup>Distribution of {crime.capitalize()} crimes by Austin neighborhoods</sup>'
                  }],
          label = crime.capitalize(),
          method = 'update'
        )
        for crime in sorted(df3_gdf['Category'].unique())
      ]),
      direction = 'down',
      showactive=True
    )
  ]
)

fig3.update_geos(
  fitbounds = 'locations',
  visible=True
)

# fig4

df4_ = df.copy()

with open('austin_neighborhoods.geojson', 'r') as file:
  austin_neighborhoods = json.load(file)
  
austin_neighborhoods_df = gpd.read_file('austin_neighborhoods.geojson')

df4_['geometry'] = df4_.apply(lambda x: Point(x['Longitude'], x['Latitude']), axis = 1)
df_v4 = df4_.copy()

df_v4 = df_v4[ (df_v4['Latitude'].isna() == False) | (df_v4['Longitude'].isna() == False)]

df_v4_gdf = gpd.GeoDataFrame(df_v4, geometry='geometry', crs='EPSG:4326')
df_v4_gdf = gpd.sjoin(df_v4_gdf, austin_neighborhoods_df, how='inner', predicate='within')

df_v4_gdf = df_v4_gdf.groupby(['neighname', 'Category', "Year", 'Location Type']).size().reset_index(name = 'Count')

neighborhoods = df_v4_gdf['neighname'].unique()
categories = df_v4_gdf['Category'].unique()
year = df_v4_gdf['Year'].unique()
loc = df_v4_gdf['Location Type'].unique()


df_v4_viz_mi = pd.MultiIndex.from_product([neighborhoods, categories, year, loc], names = ['neighname', 'Category', 'Year', 'Location Type'])
df_v4_gdf = df_v4_gdf.set_index(['neighname', 'Category', 'Year', 'Location Type'])
df_v4_gdf = df_v4_gdf.reindex(df_v4_viz_mi, fill_value=0).reset_index()

data4 = df_v4_gdf.copy()

selected_neighborhoods = ["DOWNTOWN", "WEST UNIVERSITY", "RIVERSIDE", 'UT']
#top5_neighborhoods = data.groupby('neighname')['Count'].sum().nlargest(10).index.tolist()
selected_categories = ["theft"]
#selected_location_types = ["PARKING /DROP LOT/ GARAGE"]

filtered_data4 = data4[
    (data4['neighname'].isin(selected_neighborhoods)) &
    (data4['Category'].isin(selected_categories))
]

top5_locations = filtered_data4.groupby('Location Type')['Count'].sum().nlargest(5).index.tolist()
top5_locations.append('SCHOOL - COLLEGE / UNIVERSITY')

filtered_data4 = filtered_data4[filtered_data4['Location Type'].isin(top5_locations)]
location_map = {
    "HWY / ROAD / ALLEY/ STREET/ SIDEWALK": "Street",
    "COMMERCIAL / OFFICE BUILDING": "Office",
    "PARKING /DROP LOT/ GARAGE": "Parking Lot",
    "RESIDENCE / HOME": "Residencies",
    "BAR / NIGHTCLUB": "Bar",
    'SCHOOL - COLLEGE / UNIVERSITY': 'College Campus',
    # Add others as needed...
}



df4 = filtered_data4.copy()


max_count = df4.groupby(["neighname", "Year", "Location Type"])["Count"].sum().max()

df4["Year"] = df4["Year"].astype(int)
df4["Location Clean"] = df4["Location Type"].map(location_map).fillna(df4["Location Type"])

# Add a Decade column
def get_decade(year):
    if 2000 <= year < 2010:
        return "2000s"
    elif 2010 <= year < 2020:
        return "2010s"
    elif 2020 <= year <= 2029:
        return "2020s"
    else:
        return "Other"

df4["Decade"] = df4["Year"].apply(get_decade)

# Optional: Filter out any years that didn't fall into the 3 main decades
df4 = df4[df4["Decade"].isin(["2000s", "2010s", "2020s"])]

# Update the Altair chart
fig4 = (
    alt.Chart(df4)
    .mark_bar()
    .encode(
        x=alt.X("Location Clean:N", sort=top5_locations, title=None, axis=alt.Axis(labelAngle=45)),
        y=alt.Y("sum(Count):Q", title="Crime Count", scale=alt.Scale(domain=[0, 3800])),
        color=alt.Color("Location Clean:N", title="Location Type"),
        column=alt.Column("Decade:N", title="Decade", spacing=10),
        row=alt.Row("neighname:N", title=None),
        tooltip=["neighname", "Location Clean", "sum(Count)", "Decade"]
    )
    .properties(width=100, height=150, title="Theft Count by Neighborhood, Decade, and Location Type")
)


# fig 5

df5 = df.copy()

df5['Occurred DateTime'] = pd.to_datetime(df5['Occurred DateTime'], format='%Y-%m-%d %H:%M:%S')
df5['Report DateTime'] = pd.to_datetime(df5['Report DateTime'], format='%Y-%m-%d %H:%M:%S')

df5['Occurred Month'] = df5['Occurred DateTime'].dt.month
df5['Report Month'] = df5['Report DateTime'].dt.month
df5['Occurred Day'] = df5['Occurred DateTime'].dt.day
df5['Report Day'] = df5['Report DateTime'].dt.day
df5['Occurred Day Name'] = df5['Occurred DateTime'].dt.day_name()
df5['Report Day Name'] = df5['Report DateTime'].dt.day_name()
df5['Occurred Year'] = df5['Occurred DateTime'].dt.year

df5_oy = df5.groupby(['Occurred Year']).agg('count')
df5_oy.reset_index(inplace=True)
df5_oy = df5_oy[['Occurred Year', 'Offense']]
df5_oy.drop(22, axis=0, inplace=True)

y_mean = df5_oy['Offense'].mean()
y_std = df5_oy['Offense'].std()
df5_oy['zscore'] = (df5_oy['Offense'] - y_mean) / y_std
df5_oy['zscore_abs'] = abs(df5_oy['zscore'])

df5_oy_high = df5_oy.sort_values(by = 'zscore', ascending = False).head(5)
df5_oy_low = df5_oy.sort_values(by = 'zscore', ascending=True).head(5)

lrm = LinearRegression()
X = pd.DataFrame(df5_oy['Occurred Year'])
y = df5_oy['Offense']

lrm.fit(X, y)

y_pred = lrm.predict(X)
score = lrm.score(X, y)

# import austin population data

austin_pop_df = pd.read_csv('austin_pop.csv')
austin_pop_df = austin_pop_df.sort_values(by=['Year'], ascending=True).reset_index(drop=True)
austin_pop_df.drop(22, axis=0, inplace=True)
austin_pop_df['Population'] = austin_pop_df['Population'].astype('int64')

fig5 = go.Figure()

# number of crimes
fig5.add_trace(go.Scatter(
  x = df5_oy['Occurred Year'], 
  y= df5_oy['Offense'], 
  mode='lines',  
  name="Crimes"))

# regression line
fig5.add_trace(go.Scatter(
  x=df5_oy['Occurred Year'],
  y = y_pred, 
  mode= 'lines',
  name = "Linear Model",
  opacity=0.6,
  line = dict(dash = 'dash')
))

# austin population
fig5.add_trace(go.Scatter(
  x = austin_pop_df['Year'],
  y = austin_pop_df['Population'],
  mode = 'lines',
  name = 'Austin Population', 
  yaxis = 'y2'
))

# high outliers
fig5.add_trace(go.Scatter(
  x= df5_oy_high['Occurred Year'],
  y=df5_oy_high['Offense'], 
  mode = "markers", 
  name='High Outliers',
  marker= dict(
    color = 'Red',
    size = df5_oy_high['Offense'],
    sizemode= 'area',
    sizeref = 2.*max(df5_oy_high['Offense'])/(10.**2),
    sizemin = 4
  ),
  hoverinfo="none"))

# low outliers
fig5.add_trace(go.Scatter(
  x = df5_oy_low['Occurred Year'],
  y=df5_oy_low['Offense'], 
  mode = 'markers', 
  name = 'Low Outliers',
  marker = dict(
    color = 'Green',
    size = df5_oy_low['Offense'],
    sizemode = 'area',
    sizeref = 2.*max(df5_oy_high['Offense'])/(10.**2),
    sizemin = 4
  ),
  hoverinfo="none"))

# add annotations

fig5.add_annotation(
  x = 2008,
  y = 142005,
  text = "Peak of Austin crime",
  showarrow= True,
  arrowhead = 1
)

fig5.update_layout(
  title = dict(text = f'Total Crime in Austin from 2003-2024<br><sup>Austin crime has decreased while its population has increased.<br>R^2 = {round(score, 3)}</sup>'),
  yaxis = dict(title = dict(text = 'Number of Crimes')),
  yaxis2 = dict(
    title = dict(text = 'Number of People'),
    overlaying = 'y',
    side = "right",
    autoshift = True,
    anchor = 'free',
    tickmode = "sync"),
  xaxis = dict(range = (2000, 2026)),
  legend = dict(
    yanchor = "middle",
    y = .5,
    xanchor = "left",
    x = 1.2
  ),
  width = 900,
  height = 500,
  hovermode = "x unified"
  )

# fig 6

df6 = df5.groupby(["Year", "Occurred Month"]).agg('count').reset_index()
df6 = df6[['Year', 'Occurred Month', 'Offense']]

# multiplicative decomposition
df6_md = seasonal_decompose(df6['Offense'], model = 'multiplicative', period = 12)

fig6 = make_subplots(
  rows = 4, 
  cols = 1,
  subplot_titles = ["Observed", "Trend", "Seasonal", "Residuals"]
)

# observed plot
fig6.add_trace(
  go.Scatter(
    x = df6_md.observed.index,
    y = df6_md.observed,
    mode = "lines",
    name = "Observed"
  ),
  row = 1,
  col = 1
) 

# trend plot
fig6.add_trace(
  go.Scatter(
    x = df6_md.trend.index,
    y = df6_md.trend,
    mode = "lines",
    name = "Trend"
  ),
  row = 2,
  col = 1
)

# add trend arrow 
fig6.add_annotation(
  xref = "x",
  yref = "y",
  axref = "x",
  ayref = "y",
  ax = -160,
  ay = -55,
  x = 150,
  y = df6_md.trend.iloc[150] - 1000,
  showarrow=True,
  arrowhead= 2,
  row = 2,
  col = 1
)

# seasonal plot
fig6.add_trace(
  go.Scatter(
    x = df6_md.seasonal.index,
    y = df6_md.seasonal,
    mode = "lines",
    name="Seasonal"
  ),
  row = 3,
  col = 1
)

# residual plot
fig6.add_trace(
  go.Scatter(
    x = df6_md.resid.index,
    y = df6_md.resid,
    mode = "lines", 
    name= "Residuals",
  ),
  row = 4,
  col = 1
)

# add horizontal rule to residual plot
fig6.add_hline(
  y = 0, 
  row = 4,
  col = 1,
  line_width = 2
)

fig6.update_layout(
  height = 800,
  width = 1000,
  title_text = "Multiplicative Decomposition by Month of Austin Crime from 2003-2025<br><sup>Austin crime data shows that crime is trending down and has an annual seasonality.</sup>"
)

# fig 7

df7 = pd.read_csv("apd_crime_neighborhoods.csv")
df7 = df7.drop(["Offense", "Occurred DateTime", "Report DateTime", "Census Block Group", "APD Sector", "APD District", "geometry", "index_right", "fid", "shape_area", "sqmiles", "shape_leng", "shape_length", "target_fid"], axis=1)
cat_cols = ["Time of Day", "Location Type", "neighname"]
df_encoded = pd.get_dummies(df7, columns= cat_cols)

X = df_encoded.loc[:, "Year":"neighname_ZILKER"]
y = df7['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

rf_classifier = RandomForestClassifier(max_depth=8, n_estimators=150)
rf_classifier.fit(X_train, y_train)
rfc_pred = rf_classifier.predict(X_test)
rfc_class_report = classification_report(y_test, rfc_pred, output_dict=True)
rfc_report_df = pd.DataFrame(rfc_class_report)
rfc_report_df = rfc_report_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis = 1)
rfc_report_df = rfc_report_df.transpose()
rfc_report_df.reset_index(inplace=True, names="Category")
rfc_report_df = rfc_report_df.melt(id_vars=['Category'])
rfc_report_df = rfc_report_df.rename({"variable": "Measure"}, axis = 1)

category_select = alt.selection_single(fields=['Category'], empty = 'all')

measure_dropodown = alt.binding_select(options = rfc_report_df['Measure'].unique().tolist(), name = "Measure")
measure_select = alt.selection_point(fields = ['Measure'], bind = measure_dropodown)

fig7_dist = alt.Chart(rfc_report_df).mark_bar().encode(
  alt.X('Category:N', title = 'Category'),
  alt.Y('value:Q', title = "Score"),
  alt.Color("Measure:N", scale = alt.Scale(
    domain = ["precision", "recall", "f1-score", "support"],
    range = ["#EE4B2B", "#224eb0", "#45ad69", "#a3208a" ]
  )),
  opacity = alt.condition(category_select, alt.value(1), alt.value(0.3))
).add_params(
  measure_select,
  category_select
).transform_filter(
  measure_select
).add_selection(
  category_select
).properties(
  width = 500,
  title = {
  'text' : ["Classification Report for Random Forest Classifier",
            "with max depth of eight and 150 n estimators."],
  'subtitle' : ['Poor performing because of imbalanced number of crimes for each category', 
                'Accuracy: 0.301', 
                'Macro Avg: P-0.092, R-0.085, F1-0.063', 
                'Weighted Avg: P-0.267, R-0.301, F1-0.221'],
  'anchor': 'start'
  },
)

fig7_single = alt.Chart(rfc_report_df).mark_bar().encode(
  alt.X('Measure:N', title = 'Measure'),
  alt.Y('value:Q', title = "Score"),
  alt.Color("Measure:N", scale = alt.Scale(
    domain = ["precision", "recall", "f1-score", "support"],
    range = ["#EE4B2B", "#224eb0", "#45ad69", "#a3208a" ]
  ))
).add_params(
  measure_select,
  category_select
).transform_filter(
  measure_select
).transform_filter(
  category_select
).properties(
  width = 500
)

fig7 = fig7_dist & fig7_single

st.title("Austin Crime: In Perspective")
st.header("A dashboard about Austin crime by Diego Torrealba, Alex Domond and Jenna Nega")

st.subheader("Introduction")
st.text("As people who live in Austin and go to UT, we're interested in looking at how Austin crime has evolved over the years and what important information or patterns should be noted from the data.")
st.subheader("""Visualization 1: What types of crimes are happening in Austin?""")
st.plotly_chart(fig1)
st.text("First, the overview of crimes shows that theft and burglary are the most common crime categories in Austin.")
st.divider()
st.subheader("""Visualization Two: When do these crimes happen?""")
st.altair_chart(fig2)
st.text("Theft in Austin most commonly occurs in the afternoon, while burglary is most common at night.")
st.text("Overall, however, night is the most common time of day for crime in Austin, while the morning is the least common.")
st.divider()
st.subheader("""Visualization Three: Where in Austin is crime most prevalent?""")
st.plotly_chart(fig3, theme=None)
st.text("Downtown Austin is by the far leading neighborhood in theft in Austin, while Riverside has the third most thefts. Downtown and Riverside have high counts of burgarly, but North Austin has the highest burgarly rates.")
st.text("Generally, high crime frequency across categories for Downtown, Riverside and North Lamar Rundberg.")
st.divider()
st.subheader("""Visualization Four: Location Type Deep Dives: In what locations should we be most aware?""")
st.altair_chart(fig4, theme=None)
st.text("In West Campus and Riverside, theft is most common in parking lots and at residencies, while in Downtown, most thefts occur on the street or inside of bars. Fortunately, theft is not very common on UT's campus")
st.divider()
st.subheader("""Visualization Five: Is Austin becoming safer overall?""")
st.plotly_chart(fig5, theme=None)
st.text("Overall, the count of crimes in Austin has steadily decreased since 2003 even with a population boom.")
st.text("Austin crime continually gone down since its peak at 142,000 crimes in 2008.")
st.divider()
st.subheader("""Visualization Six: Multiplicative Decomposition of Austin Crime""")
st.plotly_chart(fig6, theme=None)
st.text("This reinforces the idea that Austin crime is trending down. Additionally, there is evidence of seasonality with monthly crimes, with months having distinct average crime frequencies")
st.divider()
st.subheader("""Visualization Seven: Random Forest Classifier Classification Report""")
st.altair_chart(fig7)
st.text("The poor-performing model demonstrates the imbalanced number of observations for each category as well as not enough information to be able to confidently determine the category of crime based on available variables.")
st.divider()
st.subheader("Conclusion")
st.markdown("""
            - Theft and burglary offenses are the most common category of crime in Austin.
            - Theft has historically occurred mostly in the afternoon while burglary occurs commonly at night.
            - Downtown Austin and Riverside lead in theft and burglary crimes in Austin.
            - In West Campus and Riverside, theft is most common in parking lots and residencies.
            - In Downtown, most thefts occur on the street or inside bars. 
            - Austin crime has steadily declined since 2008 despite Austin's population increase. 
            - Austin crime exhibits a seasonality by year, which months having similar crime frequencies across years. 
            - Immense imbalances in the observances of different crime categories shows how violent crime, excluding theft and burglary, are uncommon due to how large the dataset is.
            """)