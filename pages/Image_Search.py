import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import os
from dotenv import load_dotenv

def image_search_page():
    load_dotenv()
    api_key = os.getenv("api_key")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_image_description(image_data, model, prompt="Guess this place and give a detailed description for this place"):
        image1 = {'mime_type': 'image/jpeg', 'data': image_data}
        response = model.generate_content([prompt, image1])
        return response.text
    
    def text_to_audio(text):
        tts = gTTS(text=text, lang='en')
        temp_audio_file = "temp_audio.mp3"
        tts.save(temp_audio_file)
        
        with open(temp_audio_file, "rb") as f:
            audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3")
        
        os.remove(temp_audio_file)

    # Add custom header
    st.markdown(
        """
        <h1 style='font-family: "Playfair New Zealand", serif; 
                   font-weight: 400; 
                   font-size: 50px; 
                   color: #FF6B6B; 
                   text-align: center;'>TRANQUIL TRAILS</h1>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("<p style='text-align:center; font-size: 25px;'><i>\"Uniting scenic travel and mindful cuisine to nurture body, mind, and spirit in calm locales.\"</i></p>", unsafe_allow_html=True)


    # Image uploader
    st.markdown("<p style='font-size: 18px; font-family: Poppins, sans-serif; color: #4B4B4B; margin-bottom: 20px;'>Upload an image to generate a description:</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=400)
        image_data = uploaded_file.read()
        generated_text = generate_image_description(image_data, model)
        
        st.markdown("### Generated Description")
        st.markdown(f"<p style='font-size: 18px; font-family: Poppins, sans-serif; color: #4B4B4B; margin-bottom: 20px;'>{generated_text}</p>", unsafe_allow_html=True)
        text_to_audio(generated_text)

image_search_page()
