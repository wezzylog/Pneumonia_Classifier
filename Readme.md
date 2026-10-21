# Pneumonia Classification (Streamlit)

Upload a chest X-ray and get a NORMAL / PNEUMONIA prediction with a confidence score.
For educational purposes only. Not a medical diagnosis.

## Structure
```
app.py            Streamlit UI
util.py           classify() + set_background()
train.py          trains MobileNetV2 and exports the model files below
index.html        static landing page (links to the app at localhost:8501)
models/
  pneumonia_classifier.h5
  labels.txt      "0 NORMAL" / "1 PNEUMONIA"
bgs/bg5.png       background image (optional)
requirements.txt
```

## Run
```
pip install -r requirements.txt
python train.py --data_dir chest_xray     # creates models/*
streamlit run app.py
```
Dataset: Kaggle "Chest X-Ray Images (Pneumonia)" (train/ and test/ folders).