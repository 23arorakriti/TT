import streamlit as st
from app import get_qa_chain, create_vector_db
from review import search_youtube
from gtts import gTTS
import os
from dotenv import load_dotenv


def search_page():
    load_dotenv()
    api_key = os.getenv("api_key")
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []


    def handle_query(query):
        try:
            chain = get_qa_chain()
            response = chain.invoke(query)
            return response
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return f"Error in processing the query: {str(e)}"

    def handle_youtube_search(api_key, query):
        results = search_youtube(api_key, query)
        return results

    def text_to_audio(text):
        tts = gTTS(text=text, lang='en')
        temp_audio_file = "temp_audio.mp3"
        tts.save(temp_audio_file)
        
        with open(temp_audio_file, "rb") as f:
            audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3")
        
        os.remove(temp_audio_file)

    def handle_search(query):
        create_vector_db()
        if "reddit" in query.lower() or "reviews" in query.lower():
            st.write("I don't know.")
        elif "youtube" in query.lower():
            youtube_api_key = 'AIzaSyAYvoBvpWrvCrH5QTk0NGq11p5PUMtWevc'
            try:
                results = handle_youtube_search(youtube_api_key, query)
                if results:
                    st.write(f"Top {len(results)} results for '{query}':")
                    for i, result in enumerate(results, start=1):
                        st.write(f"*Result {i}:*")
                        st.write(f"*Title:* {result['title']}")
                        st.write(f"*Description:* {result['description']}")
                        st.write(f"[Watch Video]({result['video_url']})")
                        st.image(result['thumbnail_url'])
                        st.write("---")
                    st.session_state['history'].append((query, f"YouTube results for '{query}'"))
                else:
                    st.warning("No results found.")
            except Exception as e:
                st.error(f"An error occurred during YouTube search: {e}")
                st.write("Here's the response for debugging purposes:")
                st.write(results)  # Outputting the results for debugging
        else:
            response = handle_query(query)
            st.header("Answer")
            st.markdown(f"<p style='font-size: 18px; font-family: Poppins, sans-serif; color: #4B4B4B; margin-bottom: 20px;'>{response}</p>", unsafe_allow_html=True)
            text_to_audio(response)
            st.session_state['history'].append((query, response))


   
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

    
    
    st.markdown("<p class='ask-question-text' style='font-size: 18px; font-family: Poppins, sans-serif; color: #4B4B4B; margin-bottom: 20px;'>Enter your query:</p>", unsafe_allow_html=True)
    user_query = st.text_input("", "", key="user_query")
    
    
    st.markdown(
        """
        <p style='font-size: 18px; font-family: "Poppins", sans-serif; color: #4B4B4B; margin-bottom: 20px;'>
        Examples:\n
        "Explain Spirituality of Goa in detail.",\n
        "Give a tour itinerary of Kashmir keeping in mind to experience every spiritual practice done over there",\n
        "Give some youtube vlogs for visit to Golden Temple."
        </p>
        """, 
        unsafe_allow_html=True
    )

    
    if st.button("Submit Query", key="submit_query", use_container_width=True):
        handle_search(user_query)

search_page()