# Gen-AI
This AI-powered multi-tool app includes anime-style image conversion, day/night scene translation, grammar correction, and text style transfer. Built with Streamlit and Google Gemini API, it transforms visuals and text intelligently, enhancing creativity and communication.


## 🔧 Apps Included

### 1. 🎨 Anime Style Converter
Convert real-world photos into anime-style images using image processing and cartoon filters.

### 2. 🌇 Day/Night Scene Translator
Transform scenes between day and night, complete with atmospheric effects like sun, moon, shadows, and lighting.

### 3. ✍️ Grammar Correction Tool
Automatically correct grammar, improve sentence structure, and enhance writing fluency using Google Gemini.

### 4. 🎭 Text Style Transfer
Rewrite text into various tones like professional, casual, sarcastic, poetic, and Shakespearean using AI.

---

## 🛠️ Technologies Used
- **Streamlit** – Web interface
- **Google Generative AI (Gemini)** – Text and image analysis
- **OpenCV / NumPy / PIL** – Image transformations
- **Skimage** – Exposure and gamma adjustments


## Set Up Environment
Create a .env file with:
GEMINI_API_KEY=your_google_gemini_api_key

Install dependencies:
pip install -r requirements.txt

## Run Any App
streamlit run anime.py
# or
streamlit run day.py
# or
streamlit run Grammar_Correction.py
# or
streamlit run textstyle.py
