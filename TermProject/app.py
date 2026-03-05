# Set up working libraries

import streamlit as st
from src.st_generate_output import st_generate_output_text
from st_copy import copy_button

# Set up app's name

st.set_page_config(
    page_title="FH-Translator",
    page_icon=":black_nib:"
)

# App title and brief description

st.title("Taste of Thoughts, Homes of Ideas")
st.subheader("Welcome to an Experimental Translation Tool!")
st.write("Translate your emotion :open_book: into a sensory description of food :spaghetti: or home :house:")

# Instruction

st.header("Instruction", divider=True)
st.write("1. Write down, express and describe how you feel or what you are thinking about at this particular moment (in English)")
st.write("2. Choose whether you want to translate your text into a sensory description of food ('Culinary') or home ('Interior')")
st.write("3. Click at the button 'Generate' to translate the text")

# Get an input from user and display a number of words

user_input = st.text_area("Hey! How do you feel today? What are you thinking about? Please share your feelings or thoughts with me! The more input you give, the more fruitful output I can provide :)",
                          height=220)
st.write(f"You wrote {len(user_input.split())} words.")

# Choose a translation domain

mode = st.selectbox("Choose transformation domain:",["Culinary", "Interior"])

# Generate a result

if st.button("Generate"):
    if user_input:
        with st.spinner("Generating result..."):
            result = st_generate_output_text(user_input, mode)
        if result and result.strip():  # result is not blank
            st.success("Result:")
            st.write(result)
            copy_button(
                result,
                tooltip="Copy this text",
                copied_label="Copied!",
                icon="st",
            ) # copy button from https://github.com/alex-feel/st-copy
        else:
            st.warning("No text was generated. Please click the button again.")

        # Take a Survey
        st.header("Take a Survey", divider=True)
        st.write("How is your impression with the app? If you have 5 minutes, please take this survey below")
        st.link_button("Click here to go to the survey", "https://forms.gle/pPMKhbLKg61q7Jbt6")

    else:
        st.warning("Please enter some input text before generating.")