from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
import random
import time

st.sidebar.subheader("📞 Call Me ")
st.sidebar.markdown("01050605580")
st.sidebar.markdown("📧 eng.omarfathy98697@gmail.com")
st.sidebar.markdown("my linked in account : [www.linkedin.com/in/omar-fathy-170865325]("
                    "https://www.linkedin.com/public-profile/settings?trk"
                    "=d_flagship3_profile_self_view_public_profile)")

https://github.com/Omarfathy56/chatbot.git
# Function to predict whether the image is of a cat or a dog
def predict_image(image_file):
    cnn = load_model("model.keras")

    # Load the image and preprocess it
    img = image.load_img(image_file, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    result = cnn.predict(img_array)
    if result[0][0] == 1:
        prediction = "dog"
    else:
        prediction = "cat"
    return prediction


# Function to generate typing-like response
def response_generator():
    response = random.choice(
        [
            "Hello there! Welcome to the Cat vs Dog app. Please upload a photo.",
            "Cat vs Dog is here! Upload the photo.",
            "Cat or Dog? Upload a photo, and I'll find out!",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def chatbot_message(num, var):
    list1 = [f"The uploaded image is a {var}.",
             "Uploaded an image."]

    response = list1[num]

    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def reset_conversation():
    st.session_state.conversation = None
    st.session_state.chat_history = None


user_input = "yes"

# App title
st.title("Cat vs Dog Classifier Bot")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("assistant"):
    response_text = "".join(response_generator())
    st.markdown(response_text)

uploaded_file = st.file_uploader("Upload a cat or dog image", type=["jpg", "jpeg"])
if uploaded_file:
    # Save the file in session state for consistency
    response_text1 = "".join(chatbot_message(1, None))
    st.session_state.messages.append({"role": "user", "content": response_text1})

    # Display the prediction
    with st.chat_message("assistant"):
        prediction = predict_image(uploaded_file)
        st.markdown(f"The uploaded image is a **{prediction}**!")
        response_text2 = "".join(chatbot_message(0, prediction))
        st.session_state.messages.append({"role": "assistant", "content": response_text2})
        uploaded_file = False

    with st.chat_message("assistant"):
        st.markdown("Upload another one if you want ")
