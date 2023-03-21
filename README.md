# Medical-Image-Captioning-on-Chest-X-rays
Medical imaging is the process of creating visual representations of the interior of a body for clinical analysis as well as visual representation of the function of some organs or tissues. They are widely used in hospitals and clinics to determine fractures and diseases. The medical images are read and interpreted by specialized medical professionals and their findings regarding each body of area examined are communicated via written Medical Reports. The process of writing medical reports usually takes around 5–10 minutes per report. In a day the doctors have to write medical reports that number in 100s which can take a lot of their time. The objective of this case study is to build a deep learning model that automatically write the impression part of medical report of chest X-rays and alleviate some of the burden of the medical professional. Here I will be taking a publicly available dataset from Indiana University which consists of chest X-ray images and reports (in XML format) which contain information regarding the findings and impression of the X-ray. The goal is to predict the impressions of the medical report attached to the images.

This web application takes input X-Ray images and returns the predicted impression of the medical report.

My detailed approach can be viewed in this [medium](https://towardsdatascience.com/medical-image-captioning-on-chest-x-rays-a43561a6871d) article. 


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/ashishthomaschempolil/medical-image-captioning-on-chest-x-rays/main/final.py)

![Hnet-image](https://user-images.githubusercontent.com/12584994/115959582-de0aaf00-a4da-11eb-9695-319030527a4e.gif)


# Results
| Sl No. | Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4
| - | --------------------- | ----------- | -- | -- | -- |
| 1. | Attention Model (greedy search) | 0.306819 | 0.302596 | 0.339031 | 	0.383689 |
| 2. | Custom Final Model (greedy search) | 0.214501 |	0.243265 |	0.303785 |	0.36675 |
| 3. | Simple Encoder Decoder (greedy search) | 0.317412 |	0.308454 |	0.333496 |	0.366244 |

Contents of the Code Files are given below :-

| Code File | Description  | 
| ----  | --------- |
| 1_EDA_Medical_Report.ipynb   | Exploratory Data Analysis|
| 2_Simple_encoder_decoder_Medical_Report.ipynb   | Simple Encoder Decoder Model |
| 3_Attention_Model_Medical_Report.ipynb    | Attention Model|
| 4_Custom_Final_Model.ipynb   | Model based on [Q. Tang, F. Liu, T. Zhang, J. Jiang, Y. Zhang, Attention-guided Chained Context Aggregation for Semantic Segmentation (2020)](https://arxiv.org/abs/2002.12041v3) paper|
| 5_Final.ipynb    |  	Function 1 - takes input images, returns predicted caption,Function 2 - takes input images returns BLEU scores (This file contains full data pipeline)|
| create_model.py    |  	Contains the code for creating the model|
| final.py    |  	Contains the code for web application|
| chexnet_weights   |  	Contains the weights for the ChexNet model|
| Encode_Decoder_global_attention.h5   |  	Contains the weights for the Attention Model (best model)|
| tokenizer.pkl   |  	Contains the tokenizer|

Experimentation was done in Jupyter Notebook and the code was then converted to .py files for deployment. The code for the web application is in final.py file. The web application is deployed using Streamlit.
