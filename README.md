# Disease Classifier based on symptoms

Random Forest model trained on a binary symptom matrix (132 symptoms, 41 diseases),
served through a Streamlit app. **Educational demo, not medical advice.**

## Run locally
    pip install -r requirements.txt
    python train.py          # creates model/model.joblib
    streamlit run app.py

## Deploy
- **Streamlit Community Cloud:** push to GitHub, then at share.streamlit.io choose the repo and `app.py`. Commit `model/` (it is only ~1.5 MB).
- **Hugging Face Spaces:** new Space, SDK "Docker" (or Streamlit), upload these files. Set `app_port: 8501` in the Space README header for Docker.
- **Any container host (Render, Railway, Fly.io, Cloud Run):** `docker build -t disease-clf . && docker run -p 8501:8501 disease-clf`
# disease-classifier
