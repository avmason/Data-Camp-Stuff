import pandas as pd
import numpy as np
import plotly.express as px
import warnings 
import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go
import statsmodels.api as sm

warnings.filterwarnings("ignore") 
pd.set_option('display.max_colwidth', None)


st.title('Data Science World Happiness EDA')
st.header('Target')
st.write('The target of this EDA is to draw conclusions about the factors that lead to hapiness in various regions across the globe. The information and visualizations provided are based on the "World Hapiness Data Set" as of 2021.  This will be done through the use of comprehensive charts, tables, and maps that will paint an overall picture of the given data set.')
# Data Exploration


# wh_df21 = pd.read_csv("data/clean_21.csv")
wh_df21 = pd.read_csv('data/world-happiness-report-2021.csv')
#region averages data set
gb_wh21 = wh_df21.groupby('Regional indicator', as_index=False).mean()
# Streamlit
st.write(wh_df21.head(10))

st.header('List of Features')
definitions = pd.read_csv('data/Features definition.csv')
st.write(definitions)


st.header('Data Cleaning')
st.write('Here, upon reviewing the value counts and determining how valuable certain features would be compared to others, it was decided that Standard error of ladder score, upperwhisker, lowerwhisker, Explained by: Log GDP per capita, Explained by: Social support, Explained by: Healthy life expectancy, Explained by: Freedom to make life choices, Explained by: Generosity, and Explained by: Perceptions of corruption would be dropped due to the other features being more workable for exploratory purposes.')

#Make a list of what you want to drop
columns_to_drop = ['Standard error of ladder score', 'upperwhisker', 'lowerwhisker', 'Explained by: Log GDP per capita', 'Explained by: Social support',
       'Explained by: Healthy life expectancy',
       'Explained by: Freedom to make life choices',
       'Explained by: Generosity', 'Explained by: Perceptions of corruption']

#Drop the columns using drop()
wh_df21.drop(columns_to_drop, axis=1, inplace = True) #axis = 1 lets pandas know we are dropping columns, not rows.

#Check that they are dropped
st.write(wh_df21.head(5))

missing_df = (100*wh_df21.isnull().sum()/len(wh_df21)).to_frame()
missing_df.columns = ['percentage missing']
missing_df.sort_values(by = 'percentage missing')

# st.header('Exploring Correlation')
# df_corr = df.corr() # Generate correlation matrix
# x = list(df_corr.columns)
# y = list(df_corr.index)
# z = np.array(df_corr)

st.header('Regional Map')
fig = px.choropleth(wh_df21,locations = 'Country name',
                   locationmode='country names', color= 'Regional indicator')
fig.update_layout(title = "Regional Map")
fig.update_layout(width=1000,height=1000)
st.plotly_chart(fig)

st.header('Scatter Plot')

# Healthy Life Expectancy vs. Ladder Score
st.write('Healthy Life Expectancy vs. Ladder Score')
st.write('This scatterplot visualization shows positive, medium/strong correlation. In general, the higher the healthy life expectancy, the higher the ladder score & the happier the general country is. However, there are a few of outliers where there is a high healthy life expectancy but a relatively lower ladder score.')
fig_hle_vs_ls = px.scatter(wh_df21, x = 'Healthy life expectancy', y = 'Ladder score', color = 'Country name', title = 'Healthy Life Expectancy vs. Ladder Score')
val_x = 'Healthy life expectancy'
val_y = 'Ladder score'
model = sm.OLS(wh_df21[val_y], wh_df21[val_x]).fit()
fitted_vals=model.fittedvalues
fig_hle_vs_ls.add_trace(go.Scatter(x=wh_df21[val_x],
                         y=fitted_vals,
                         mode='lines',
                         name='best fit',
                         line=dict(color='firebrick', width=2)
                        ))
st.plotly_chart(fig_hle_vs_ls)

# freedom to make life choices vs. perceptions of corruption
fig = px.scatter(wh_df21, x="Freedom to make life choices", y="Perceptions of corruption", color="Country name", title="Freedom vs Corruption by Location & Dystopia", size="Dystopia + residual")
st.write('Freedom to Make Life Choices vs. Perceptions of Corruption')
st.write('There is no clear linear correlation between freedom to make life choices and perceptions of corruption. While regions with low ratings for freedom to make life choices usually have high perceptions of corruption, regions with high ratings for freedom to make life choices have both high and low perceptions of corruption.')

st.plotly_chart(fig)


#scatter plot gdp vs. life expectancy
st.write('GDP vs. Life Expectancy')
fig = px.scatter(wh_df21, x="Logged GDP per capita", y="Healthy life expectancy",  color="Regional indicator")
st.write('Sub-Saharan African countries tend to have a lower healthy life expectancy and logged GDP per capita, whereas Western European countries predominantly have the highest. There is a positive correlation between the healthy life expectancy and logged GDP per capita of regions.')
val_x = "Logged GDP per capita"
val_y = "Healthy life expectancy"
model = sm.OLS(wh_df21[val_y], wh_df21[val_x]).fit()
fitted_vals=model.fittedvalues
fig.add_trace(go.Scatter(x=wh_df21[val_x],
                         y=fitted_vals,
                         mode='lines',
                         name='best fit',
                         line=dict(color='firebrick', width=2)
                        ))
st.plotly_chart(fig)

#scatter plot gdp vs. generosity         
fig = px.scatter(wh_df21, x="Logged GDP per capita", y="Generosity", size="Healthy life expectancy", color="Regional indicator")
st.write('GDP vs. Generosity')
st.write('There is no clear trend demonstrated between the generosity and logged GDP per capita of countries.')
val_x = "Logged GDP per capita"
val_y = "Generosity"
model = sm.OLS(wh_df21[val_y], wh_df21[val_x]).fit()
fitted_vals=model.fittedvalues
fig.add_trace(go.Scatter(x=wh_df21[val_x],
                         y=fitted_vals,
                         mode='lines',
                         name='best fit',
                         line=dict(color='firebrick', width=2)
                        ))
st.plotly_chart(fig)

#scatter matrix
numerical_wh_df21 = wh_df21[['Logged GDP per capita', 'Healthy life expectancy', 'Ladder score']]
fig = px.scatter_matrix(numerical_wh_df21, color='Healthy life expectancy')
st.write('Scatter Matrix of GDP, Life Expectancy, and Ladder Score')
st.write('There is a positive trend for all the graphs, indicating that Healthy life expectancy, Ladder score, and Logged GDP per capita depend on one another.')
st.plotly_chart(fig)

st.header('Correlation Heatmap')

fig = px.imshow(wh_df21[wh_df21.columns[wh_df21.columns != 'Ladder score in Dystopia']].corr(), text_auto=True)
st.write('An important column to focus on is the ladder score because ultimately this shows the happiness of a country. It is apparent here that there is a high correlation of happiness between Logged GDP per capita, Social support, Healthy Life Expectancy, and Freedom to make choices as these all have correlations around 0.7. Additionally it can be spotted how generosity stand in the lowest values with a score of -0.017')
fig.update_layout(width=1000,height=800)
st.plotly_chart(fig)


st.header('')


st.header('Bar Graphs')
avg_wh21 = wh_df21.groupby('Regional indicator', as_index=False).mean()
std_wh21 = wh_df21.groupby('Regional indicator', as_index=False).std()

# logged GDP per capita by region with standard deviation & errors
fig = go.Figure()
fig.add_trace(go.Bar(
    name='Control',
    x=avg_wh21['Regional indicator'], y=avg_wh21['Logged GDP per capita'],
    error_y=dict(type='data', array=std_wh21['Logged GDP per capita']),
    marker_color = px.colors.qualitative.Plotly
))
st.write('Average Logged GDP per Capita by Region with Standard Deviation & Errors')
st.write('The following error bar chart shows in which range the majority of the average logged GDP per capita by region is. For example, North America and ANZ has the smallest variation of 0.16 among all the regions, with 68% of the data logged GDP being around 10.8, while Southeast Asia has the largest variation of 0.97, with 68% of its data logged GDP being around 9.4.')
st.plotly_chart(fig)

# healthy life expectancy by region
fig = px.bar(avg_wh21, x='Regional indicator', y='Healthy life expectancy', color = 'Regional indicator', title='Healthy Life Expectancy by Region')
st.write('Average Healthy Life Expectancy by Region')
st.write('In general, the average healthy life expectancy for all the regions provided in this data set is in the approximate range of 50 to 80 years. The maximum average regional life expectancy is  about 73 years, in the Western Europe region. And the minimum average regional life expectancy is about 56 years, in sub-Saharan Africa.')
st.plotly_chart(fig)

# generosity by region with standard deviation & errors
px.bar(avg_wh21, x='Regional indicator', y='Generosity', color = 'Regional indicator', title='Generosity by Region')
fig = go.Figure()
fig.add_trace(go.Bar(
    name='Control',
    x=avg_wh21['Regional indicator'], y=avg_wh21['Generosity'],
    error_y=dict(type='data', array=std_wh21['Generosity']),
    marker_color = px.colors.qualitative.Plotly
))
fig.update_layout(barmode='group')
st.write('Average Generosity by Region with Standard Deviation & Errors')
st.write('The average generosity can vary from a bit below 0 to a bit above zero for each region. However, the variety within each region, except from North America, is very different, as majority of the data is way above 0. The majority of the data from all other regions has a greater range that extends to both negative and positive generosity.')
st.plotly_chart(fig)

st.header('Pie Charts')
#countries per regional indicator
count_wh_df21 = wh_df21.groupby('Regional indicator', as_index=False).count()
fig = px.pie(count_wh_df21, values='Country name', names= 'Regional indicator', title='Countries per Regional Indicator')
st.write('Countries per Regional Indicator')
st.write('It is illustrated that Sub-Saharan Africa has the most countries per region, and North America and ANZ have the least countries per region.')
st.plotly_chart(fig)

#pie chart for average ladder score per region
fig = px.pie(gb_wh21, values='Social support', names= 'Regional indicator', title='Average Social Support per Region')
st.write('Average Social Support per Region')
st.write('It is demonstrated that North America and ANZ and Western Europe have the highest levels of social support, whereas South Asia and Sub-Saharan Africa have the lowest levels of social support compared to other regions.')
st.plotly_chart(fig)

st.header('World Map')
fig = px.choropleth(wh_df21,locations = 'Country name',
                   locationmode='country names', color= 'Ladder score')
fig.update_layout(title = "World Happiness")
st.write('To continue our EDA visualizations, we provided the distribution of the given Ladder score in estimated countries, with the conclusion that North America and Western Europe are among the happiest regions, followed by Latin America.')
fig.update_layout(width=1000,height=800)
st.plotly_chart(fig)

fig = px.scatter_geo(wh_df21,locations = 'Country name', size = 'Ladder score',
                   locationmode='country names', color= 'Social support')
fig.update_layout(title = "Social Support")
st.write('Countries with the highest ladder score are generally found on three continents: North America, Europe, and ANZ (Australia-New Zealand). Countries with the lowest ladder score are generally found on Africa, while countries with a moderate ladder score are generally found on Asia and South Africa.')
fig.update_layout(width=1000,height=800)
st.plotly_chart(fig)

fig = px.scatter_geo(wh_df21,locations = 'Country name', size = 'Ladder score',
                   locationmode='country names', color= 'Logged GDP per capita')
fig.update_layout(title = "Logged GDP per capita")
fig.update_layout(width=1000,height=800)
st.write('Countries with the highest social support are generally found on three continents: Europe, North America, and South America but are also present on Asia and ANZ. Countries with highest social support may also have highest ladder scores.')

st.plotly_chart(fig)
fig = px.scatter_geo(wh_df21,locations = 'Country name', size = 'Ladder score',
                   locationmode='country names', color= 'Healthy life expectancy')
fig.update_layout(width=1000,height=800)
fig.update_layout(title = "Healthy life expectancy")
st.write('Europe is a leader in the highest GDP per capita, signifying high economic stability. The Middle East, ANZ, and Eastern Asia are also notably wealthy regions, whereas Africa primarly has the lowest GDP per capita. Regions with more established GDP levels are likelier to have improved life expectancies and quality of life, effectuating higher ladder scores.')
st.plotly_chart(fig)
st.write ('Western Europe, Eastern Asia, and ANZ have the highest healthy life expectancies. Conversely, African nation have the lowest life expectancies. Such results could be due to high economic development, which generates advancements towards modern medicine and public health. Developing countries are more prone to having lower healthy life expectancies due to less financial support for government healthcare programs.')

st.header('Violin Plots')

fig = px.violin(wh_df21, x='Regional indicator', y="Logged GDP per capita", title="Regional indicator vs. GDP per capita")
st.write('The graph below represents the GDP trend in regions across the globe. According to the graph, we can see that Western Europe and the North America ANZ are consistently the most happy. The middle east on the other hand has a a wide variety of different happiness levels with subsaharan Africa having a consistently high amount of unhappiness.  ')
st.write('Violin Plot GDP and Regional Indicator')
st.plotly_chart(fig)

fig = px.violin(wh_df21, x='Regional indicator', y="Perceptions of corruption", title="Regional indicator vs. Perceptions of corruption")
st.write('The graph below displays perceptions of corruption in different areas. WesternEurope has a wide range of perceptions along with North America. The areas in which people feel most secure ')
st.write('Violin Plot corruption and Regional Indicator')
st.plotly_chart(fig)

fig = px.violin(wh_df21, x='Regional indicator', y="Social support", title="Regional indicator vs. Social support")
st.write('There is not a significant difference in the trend shown here ')
st.write('Violin Plot Regional Indicator vs. Social Support')
st.plotly_chart(fig)

st.header('Drawing Conclusions')
st.write('Based on the exploration performed, we can conclude a few things. It is evident that countries with a higher GDP per capita tend to have higher healthy life expectancies and ladder scores. However, the data does not suggest that higher GDP per capita correlates to increased levels of generosity within a nation. Citizens with more freedom to make life choices generally have fewer perceptions of corruption. Furthermore, Western Europe and North America and ANZ are the happiest regions from their continuous preeminence in key categories, including social support and healthy life expectancy. When analyzing data based on regional indicators, it is important to consider region sizes. Sub-Saharan Africa accounted for nearly a quarter of all analyzed countries, while North America and ANZ accounted for under 5%.')
st.write('')
