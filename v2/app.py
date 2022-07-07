# load libraries
import matplotlib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from utils import (
    generate_wordcloud, preprocess_text, 
    preprocess_dataset, load_embedding, load_model,
    preprocess_only
    )
from sklearn.metrics import classification_report

# setup dashboard
st.set_page_config(page_title = "Sentiment", page_icon = "🥴", layout = "wide")

# interface section
st.title("Sentiment Analytic Dashboard with SVM & Naive Bayes")

# define mapping
id_to_label = {
    '0' : "Negatif",
    "1" : "Netral",
    "2" : "Positif"
}

# load model
model     = load_model()
trf, cvt  = load_embedding()
# embedding = lambda x, mode: trf.fit_transform(cvt.fit_transform(text)).toarray() if mode == "text" else trf.fit_transform(cvt.fit_transform(text)).toarray()

selection = st.sidebar.selectbox("Input Type :", ['Text', 'Dataset'])

if selection == "Text":
    text_input = st.text_input("", placeholder = "input text here...")
    isClick    = st.button("Predict")
    if text_input != "" and isClick:
        # text preprocessing
        text = preprocess_text(text_input)
        # prediction
        array  = cvt.fit_transform([text]).toarray()
        result = str(model.predict(array)[0]) 
        # result
        st.success(f"Detected as {id_to_label[result]} Text")

if selection == "Dataset":
    file_input = st.sidebar.file_uploader("", type = ["csv", "xlsx"])
    if file_input is not None:
        data = pd.read_csv(file_input)
        st.dataframe(data)
        _st1, _st2, _st3 = st.columns(3)
        column     = _st1.text_input("", placeholder = "input column name here...")
        size       = _st2.text_input("", placeholder = "input size row here...")
        prev_label = _st3.text_input("", placeholder = "input column label here...")
        st1, st2, st3, _  = st.columns([1, 1, 1, 6])
        isVisual  = st1.button("Visualize")
        isPredict = st2.button("Predict")
        isClean   = st3.button("Preprocess")
        if isVisual and column != "":
            try:
                # wordcloud visualisation 
                st.pyplot(generate_wordcloud(data[column]))
            except (KeyError, TypeError) as E:
                st.error(E)
        if isPredict and column != "" and size != "":
                size = int(size)
                # sentiment analytic 
                try:
                    with st.spinner("Please wait, stemming start..."):
                        text_clean = preprocess_dataset(data[column][:size])
                        list_text = cvt.fit_transform(text_clean).toarray()
                        list_predict = model.predict(list_text)
                        
                        st_1, st_2 = st.columns(2)
                        with st_1:
                            st.subheader("Pie Chart Distribution")
                            fig, ax = plt.subplots(figsize = (5, 5))
                            ax = pd.Series([id_to_label[str(i)] for i in list_predict]).value_counts().plot(kind = "pie", autopct = "%.2f%%")
                            st.pyplot(fig)
                        with st_2:
                            st.subheader("Dataframe Prediction")
                            dataset_result = pd.DataFrame({
                                "text" : text_clean, 
                                "prev_label" : data[prev_label][:size],
                                "pred_label" : [id_to_label[str(i)] for i in list_predict]})
                            st.dataframe(dataset_result)
                        
                except (KeyError, TypeError) as E:
                    st.error(E)
        if isClean and column != "" and size != "":
            size = int(size)
            # cleaning text
            try:
                with st.spinner("Please wait, preprocessing start..."):
                    text_clean = preprocess_only(data[column][:size])
                    st.dataframe(text_clean)
            except (KeyError, TypeError) as E:
                st.error(E)
        