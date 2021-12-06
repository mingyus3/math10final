#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  30 20:16:35 2021

@author: shimingyu
"""

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from tensorflow import keras
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns

st.title("Analyze COVID-19 Vaccination Data in California")
st.markdown('presented by Mingyu Shi, https://github.com/mingyus3?tab=repositories',unsafe_allow_html=True)


st.subheader('Introduction:')
st.write("In this project, I will use the vaccination data of California counties from CHHS Open Data to analyze the trends of vaccination.")



st.subheader('Main Part:')
df = pd.read_csv("covid19vaccinesbycounty.csv",na_values = " ")
st.write("Let's see the original dataset first")
st.write(df)

st.write("I want to look at vaccine data for every county in California, so I'm going to do some tweaks to this chart to get rid of the excess data.")
st.write("Now I invite you to explore this data set first.")

df = df[df.notna().all(axis=1)].copy()

df = df[df["california_flag"].str.contains("Not in California")==False]

s = st.slider(("number of rows selected"),0,len(df),[0,len(df)-1])
st.write(f"The range of the row you choose is {s}")

def can_be_numeric(c):
    try:
        pd.to_numeric(df[c])
        return True
    except:
        return False
df1 = df.applymap(lambda x: np.nan if x == " " else x)
    
        
good_cols = [c for c in df1.columns if can_be_numeric(c)]
df1 = df1[good_cols].apply(pd.to_numeric, axis = 0)



a = st.selectbox("choose a x value", good_cols)
b = st.selectbox("choose a y value", good_cols)
    
r = list(s)
    
plotrange = range(r[0],r[1])
    
df1 = df1.iloc[plotrange]



c = alt.Chart(df1).mark_circle().encode(
    x = str(a),
    y = str(b),
    tooltip = [str(a),str(b)]
    
)


st.altair_chart(c, use_container_width=True)

st.write("In fact, it's hard to get much useful information out of the raw data because it contains time and is not reprocessed for county classification.")
st.write("Since I live in Orange County, I will focus on the specific situation of this County.")
st.write("However, I will list the county data below for you to choose to see.")
dfx = df.drop_duplicates(subset =["county"])
cty = st.selectbox("choose a conty's data to see", dfx["county"])
dfx_ = df[df['county'].str.contains(cty)]
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

def highlight_min(y):
    is_min = y == y.min()
    return ['background-color: blue' if v else '' for v in is_min]


dfx_ = dfx_.style.apply(highlight_max, subset=['total_doses', 'pfizer_doses', 'moderna_doses', 'jj_doses', 'partially_vaccinated', 'fully_vaccinated', 'at_least_one_dose'] )\
                 .applymap(lambda v: 'opacity: 20%;' if (v == 0) else None, subset=['total_doses', 'pfizer_doses', 'moderna_doses', 'jj_doses', 'partially_vaccinated', 'fully_vaccinated', 'at_least_one_dose'])\
                 .apply(highlight_min, subset=['total_doses', 'pfizer_doses', 'moderna_doses', 'jj_doses', 'partially_vaccinated', 'fully_vaccinated', 'at_least_one_dose'] )

    
st.write(dfx_)
st.write("I've marked the maximum and minimum values of some columns in yellow and blue, and I've blurred out all the zeros.")
df2 = df[df['county'].str.contains('Orange')]
st.write("The rest of the data is for Orange County.")

st.write("Next take a closer look at vaccination trends in OC. And to fit it with a straight line.")
chart = alt.Chart(df2).mark_circle().encode(
    x = "cumulative_total_doses",
    y = "cumulative_fully_vaccinated",
    tooltip = ["administered_date","cumulative_total_doses"]
)
chart + chart.transform_regression("cumulative_total_doses", "cumulative_fully_vaccinated").mark_line()


st.write("With Seaborn, we can visually see the daily injection volume changes.")
g = sns.relplot(x="administered_date", y="total_doses", data=df2)
fig, ax = plt.subplots()
st.pyplot(g)

st.write("Next, we analyzed the association between the cumulative use data of the three different kinds of vaccines and the number of fully vaccinated people in OC using sklearn LinearRegression.")
df3 = df2[["cumulative_pfizer_doses","cumulative_moderna_doses","cumulative_jj_doses"]]
st.write(df3)
reg = LinearRegression()
X = np.array(df3)
y = np.array(df2["cumulative_fully_vaccinated"]).reshape(-1,1)
reg.fit(X,y)
reg.coef_
reg.intercept_
st.write(f"The result of the coefficients is {reg.coef_}")
st.write(f"The result of the y-intercept is {reg.intercept_}")



st.write("Next, use Keras to continue the exploration.")
numeric_cols = [c for c in df2.columns if is_numeric_dtype(df[c])]
scaler = StandardScaler()
scaler.fit(df2[numeric_cols])
df2[numeric_cols] = scaler.transform(df2[numeric_cols])

X_train = df2[numeric_cols].drop("cumulative_fully_vaccinated", axis=1)
y_train = df["cumulative_fully_vaccinated"]
X_train.shape
model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape = (13,)),
        keras.layers.Dense(100, activation="sigmoid"),
        keras.layers.Dense(100, activation="sigmoid"),
        keras.layers.Dense(1,activation="selu")
    ]
)

model.compile(
    loss="mean_squared_error", 
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    
)

history = model.fit(X_train,y_train,epochs=10, validation_split = 0.2)

fig, ax = plt.subplots()
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(['train', 'validation'], loc='upper right')
model.summary()
st.pyplot(fig)
st.write("In fact, each refresh results in a different image. Sometimes the train results are not good.")


st.subheader('References link:')
st.write("https://data.chhs.ca.gov/dataset/vaccine-progress-dashboard/resource/130d7ba2-b6eb-438d-a412-741bde207e1c")
st.write("https://pandas-docs.github.io/pandas-docs-travis/user_guide/style.html")
st.write("https://stackoverflow.com/questions/67869034/styler-object-has-no-attribute-style")
st.write("https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html")
st.write("https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Styler-Functions")
st.write("https://www.realpythonproject.com/how-to-use-seaborn-for-data-visualization/")
st.write("https://stackoverflow.com/questions/60474039/tooltips-in-stacked-altair-vega-lite-chart-disappear-when-using-interval-selecti")
st.write("https://stackoverflow.com/questions/63011282/how-to-display-image-from-a-folder-using-button-with-streamlit")
st.write("https://docs.streamlit.io/library/api-reference/media/st.image")


from PIL import Image
meme = st.checkbox("Hit me!")
image = Image.open('meme.JPG')
if meme:
    st.image(image, caption='What it looks like when I type code')

