import streamlit as st
import os

def load_chapter(chapter_file):
    with open(os.path.join('chapters', chapter_file), 'r') as file:
        return file.read()

chapters = {
    "Cover Page": "cover.jpg",
    "Introduction": "intro.html",
    "Gravity": "gravity.html",
    "Electrical Resistivity Tomography (ERT)": "resis.html",
    "Seismic Exploration": "seismic.html",
    "Ground Penetrating Radar": "GPR.html"
}

chapter_selection = st.sidebar.selectbox("Select a Chapter", list(chapters.keys()), index=0)

if chapter_selection == "Cover Page":
    st.image(os.path.join('chapters', chapters[chapter_selection]), use_column_width=True)
else:
    chapter_content = load_chapter(chapters[chapter_selection])
    st.markdown(chapter_content, unsafe_allow_html=True)
