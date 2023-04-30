import streamlit as st
import pandas as pd
n=4

listof_d = [{"asdasd":"asdasd"},{"aaaaaaa":" "},{"asdasd":"asdasd"}]


n=3
i=0
df = pd.DataFrame(
    [
        listof_d
   ]
)
edited_df = st.experimental_data_editor(df)
df.loc[len(df.index)] = ["asdas",22,11]

edited_df = st.experimental_data_editor(df)
