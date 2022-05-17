#---------------------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------------------
import os
import pathlib
import time


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import tensorflow as tf

from PIL import Image
from yaml import load, Loader, dump
from sklearn.decomposition import PCA
from skimage.transform import resize
from skimage.io import imread


#---------------------------------------------------------------------------------------
# Loading files
#---------------------------------------------------------------------------------------
yaml_file = open("notebooks/app.yaml", 'r')
yaml_content = load(yaml_file, Loader=Loader)

evaluation_df = pd.read_csv('planes/models/evaluation_df.csv', header=0, names=["Model", "Input shape", "Accuracy", "Precision", "Recall"])


#---------------------------------------------------------------------------------------
# Constantes
#---------------------------------------------------------------------------------------
MODELS_DIR = pathlib.Path(yaml_content["MODELS_DIR"])

TARGET_NAME = yaml_content["TARGET_NAME"]
TARGET_NAME_TXT = f'images_{TARGET_NAME}_train.txt'



PATH_MODEL = 'planes/models/' + TARGET_NAME + '.h5'
PATH_CLASSES = 'planes/models/' + TARGET_NAME + '_classes.txt'

IMAGE_WIDTH = yaml_content["IMAGE_WIDTH"]
IMAGE_HEIGHT = yaml_content["IMAGE_HEIGHT"]
IMAGE_DEPTH = yaml_content["IMAGE_DEPTH"]



list_models =[]
for model in os.listdir('planes/models'):
    if '.' in model:
        list_models.append(model)
        
classes_names_list = []
with open(PATH_CLASSES, 'r') as f:
    for name in f.readlines():
        classes_names_list.append(name.replace('\n', ''))


#---------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------
def load_image(path):
    """Load an image as numpy array
    """
    return plt.imread(path)

@st.cache(ttl = 12*3600, allow_output_mutation=True, max_entries=5)
def load_model(path, type_model):
    """Load tf/Keras model for prediction
    Parameters
    ----------
    path (Path): Path to model
    type_model (int): type of model to use for prediction
                    0 Keras model
                    1 SVM
    Returns
    -------
    Predicted class
    """
    if type_model ==0:
      return tf.keras.models.load_model(path)
    elif type_model ==1:
      return pickle.load(open(path,'rb'))


def predict_image(path, model, path_classes, type_model):
    """Predict plane identification from image.
    
    Parameters
    ----------
    path (Path): Path to image to identify
    model (keras.models / SVM): Keras model / SVM  to be used for prediction
    type_model (int): type of model to use for prediction
                    0 Keras model
                    1 SVM
    pca (pca model): model used for pca to reduce dimension for svm
    Returns
    -------
    Predicted class, prob
    """
    names = pd.read_csv(path_classes, names=['Names'])
    if type_model ==0:
        image_resized = [np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))]
        prediction_vector = model.predict(np.array(image_resized))
        predicted_classes = np.argmax(prediction_vector, axis=1)[0]
        predicted_prob = prediction_vector[0][predicted_classes] * 100
        name_classes = names['Names'][predicted_classes]
        return predicted_classes, predicted_prob, prediction_vector,name_classes
    elif type_model ==1:
        name_model_pca = [x for x in list_models if "pca" in x][0]
        PATH_MODEL_PCA = f'planes/models/{name_model_pca}'
        model_pca = load_model(PATH_MODEL_PCA, type_model)
        prob = model.predict_proba(model_pca.transform(pd.DataFrame(np.array([(resize(imread(path, plugin='matplotlib'),\
                                          (IMAGE_WIDTH, IMAGE_HEIGHT))).flatten()]))))
        predicted_classes = np.argmax(prob)
        predicted_prob = prob[0][predicted_classes]
        name_classes = names['Names'][predicted_classes]
        return predicted_classes, predicted_prob, name_classes




#---------------------------------------------------------------------------------------
# Config page
#---------------------------------------------------------------------------------------
img = Image.open("planes/app/airbus-min.png")
st.set_page_config(
     page_title="Plan classification App",
    #  page_icon=":shark",
     page_icon=img,
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://github.com/MerylAhounou/plane_classification_projet',
         'Report a bug': "https://github.com/MerylAhounou/plane_classification_projet",
         'About': "# This is an *plane classification* cool app! Let's try!!"
     }
 )



#---------------------------------------------------------------------------------------
# Head app
#---------------------------------------------------------------------------------------
# st.title("Identifation d'avion")
html_temp = """
<div style="background-color:blue;padding:1.5px">
<h1 style="color:white;text-align:center;">Identification d'avion </h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)
st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)
st.info(f'Liste des différentes classes d\'avion possible à prédire: {classes_names_list}')


#---------------------------------------------------------------------------------------
# Upload image
#---------------------------------------------------------------------------------------
uploaded_files = st.file_uploader("Charger une image d'avion") #, accept_multiple_files=True)

if uploaded_files:
    loaded_image = load_image(uploaded_files)
    st.image(loaded_image)
    
    
#---------------------------------------------------------------------------------------
# Button
#---------------------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
  add_radio = st.radio(
          "Quel modèle voulez-vous utilisez?",
      ("CNN neural network", "VGG19")
#       ("CNN neural network", "SVM", "Xception", "VGG19")
      )
  predict_btn = st.button('Identifier', disabled=(uploaded_files is None))
  

with col2:
  st.write('Performances des modèles sur le jeu de données d\'entrainement')
  st.dataframe(evaluation_df)
  prob_btn = st.button('Afficher les probabilités', disabled=(uploaded_files is None or add_radio =="SVM"))



#---------------------------------------------------------------------------------------
# Loading model
#---------------------------------------------------------------------------------------
if add_radio =="CNN neural network":
  name_model = [x for x in list_models if "cnn" in x][0]
  type_model = 0
elif add_radio =="SVM":
  name_model = [x for x in list_models if "svm" in x][0]
  type_model = 1
elif add_radio =="Xception":
  name_model = [x for x in list_models if "xception" in x][0]
  type_model = 0
elif add_radio =="VGG19":
  name_model = [x for x in list_models if "vgg19" in x][0]
  type_model = 0

PATH_MODEL = f'planes/models/{name_model}'

model = load_model(PATH_MODEL, type_model)

#---------------------------------------------------------------------------------------
# Prediction
#---------------------------------------------------------------------------------------

if type_model ==1:
  with col1:
    if predict_btn:
        my_bar = st.progress(0)
        for percent_complete in range(100):
              time.sleep(0.05)
              my_bar.progress(percent_complete + 1)
        prediction_classes, predicted_prob, prediction_names = predict_image(uploaded_files, model, PATH_CLASSES, type_model)
        st.write(f"C'est un avion de type: {prediction_names} de classes {prediction_classes} avec une\
                probabilité de prédiction de:{predicted_prob: .2f}%")
        st.balloons()
      
elif type_model !=1:
  with col1:
    if predict_btn:
        my_bar = st.progress(0)
        for percent_complete in range(100):
              time.sleep(0.05)
              my_bar.progress(percent_complete + 1)
        prediction_classes, predicted_prob, prediction_vector, prediction_names = predict_image(uploaded_files, model, PATH_CLASSES, type_model)
        st.write(f"C'est un avion de type: {prediction_names} de classes {prediction_classes} avec une\
                probabilité de prédiction de:{predicted_prob: .2f}%")
        st.balloons()
  
  with col2:
    if prob_btn:
        my_bar = st.progress(0)
        for percent_complete in range(100):
              time.sleep(0.05)
              my_bar.progress(percent_complete + 1)
        prediction_classes, predicted_prob, prediction_vector,prediction_names = predict_image(uploaded_files, model, PATH_CLASSES, type_model)
        prediction_vector = prediction_vector*100
        chart_data = pd.DataFrame(prediction_vector, columns= classes_names_list).T
        st.bar_chart(chart_data)
        st.balloons()

#---------------------------------------------------------------------------------------
# Tail
#---------------------------------------------------------------------------------------
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<b><div align="center">**``AHOUNOU Méryl et KEVORKIAN Amandine``**</div></b>',
            unsafe_allow_html=True)
