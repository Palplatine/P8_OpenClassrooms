import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

df = px.data.iris()


fig = go.Figure(data=[go.Table(header=dict(values=list(df.columns),
                align='left'),
    cells=dict(values=[df.sepal_length, df.sepal_width, df.petal_length, df.petal_width, df.species, df.species_id],
               align='left'))
])

fig1 = px.scatter(df, x="sepal_width", y="sepal_length", color="species", size='petal_length', trendline="ols", hover_data=['petal_width'])

fig2 = px.violin(df, y="sepal_width", color="species_id", box=True, points="all", hover_data=df.columns)

fig3 = px.parallel_coordinates(df, color="species_id", color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)

fig4 = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width', color='species')

st.title("The Iris dataset cause we've never seen it before, it's brand new")

st.subheader('Our dataset:')
st.plotly_chart(fig, use_container_width=True)

st.subheader('Scatter plot with OLS lines')
st.plotly_chart(fig1, use_container_width=True)

st.subheader('Violin plot')
st.plotly_chart(fig2, use_container_width=True)

st.subheader('Parallel coordinates')
st.plotly_chart(fig3, use_container_width=True)

st.subheader('Scatter plot (in 3D!)')
st.plotly_chart(fig4, use_container_width=True)