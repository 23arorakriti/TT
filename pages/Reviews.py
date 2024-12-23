import streamlit as st
import asyncio
import asyncpraw
from dotenv import load_dotenv
import os

# Async function to fetch Reddit data
async def fetch_reddit_data(query):
    async with asyncpraw.Reddit(client_id='vaGRzJF8B7WelHKQDzvf3w',
                                client_secret='Ngwb0UqRiPbnLEZQdaCWEDGrwUbLzQ',
                                user_agent='test by u/Huge_Donut9460') as reddit:

        # Define the list of subreddits and limits
        subreddits = ['india', 'culture', 'spirituality', 'travel']  # Add more subreddits as needed
        post_limit = 5

        # Initialize a list to hold URLs of posts
        all_urls = []

        for subreddit_name in subreddits:
            # Fetch posts
            subreddit = await reddit.subreddit(subreddit_name)

            try:
                async for submission in subreddit.search(query, limit=post_limit):
                    all_urls.append(submission.url)
            except asyncpraw.exceptions.RedditAPIException as e:
                if e.error_type == 'SUBREDDIT_NOT_FOUND':
                    st.warning(f"Subreddit '{subreddit_name}' not found.")
                else:
                    raise e

        # Return the list of URLs
        return all_urls

# Function to run the async function using asyncio.run()
def run_fetch(query):
    return asyncio.run(fetch_reddit_data(query))

# Streamlit interface
def reddit_search_page():
    load_dotenv()

    # Add custom header
    st.markdown(
        """
        <h1 style='font-family: "Playfair New Zealand", serif; 
                   font-weight: 400; 
                   font-size: 50px; 
                   color: #FF6B6B; 
                   text-align: center;'>Tranquil Trails</h1>
        """, 
        unsafe_allow_html=True
    )

    st.markdown("<p style='text-align:center; font-size: 25px;'><i>\"Uniting scenic travel and mindful cuisine to nurture body, mind, and spirit in calm locales.\"</i></p>", unsafe_allow_html=True)

    # Description
    st.markdown(
        "<p style='font-size: 18px; font-family: Poppins, sans-serif; color: #4B4B4B; margin-bottom: 20px;'>"
        "Enter a search query to fetch relevant Reddit posts from selected subreddits:</p>",
        unsafe_allow_html=True
    )
    

    # Input field
    query = st.text_input("", placeholder="Enter search query...", key="user_query", 
                          help="Type the topic you'd like to search on Reddit")

    # Search button
    if st.button("Search Reddit", key="submit_query", use_container_width=True):
        if query:
            urls = run_fetch(query)
            if urls:
                st.markdown("<h3 style='font-family: Poppins, sans-serif; color: #FF6B6B;'>Top Post URLs:</h3>", unsafe_allow_html=True)
                for url in urls:
                    st.markdown(f"<a href='{url}' target='_blank' style='font-size: 16px; font-family: Poppins, sans-serif; color: #4B4B4B;'>{url}</a>", unsafe_allow_html=True)
            else:
                st.warning("No posts found for the query.")
        else:
            st.warning("Please enter a search query.")

# Run the Streamlit app
reddit_search_page()
