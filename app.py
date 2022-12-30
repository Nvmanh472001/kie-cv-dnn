import streamlit as st
from PIL import Image
from numpy import asarray
from pipeline import pipeline_handle

def main():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center; color: white;'>Trích xuất CV</h1>",
                unsafe_allow_html=True)

    image_file = st.file_uploader('CONVERT TO DIGITAL IMAGE', type=[
                                  'jpeg', 'jpg', 'jpe', 'png', 'bmp'], key=1)
    if 'image' not in st.session_state:
        st.session_state.image = None

    if image_file is not None:
        img = asarray(Image.open(image_file))
        st.button("Extract Information", on_click=pipeline_handle, args=[img])
        if st.session_state.image is not None:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col2:
                st.image(st.session_state.image, width=960)

if __name__ == "__main__":
    main()
