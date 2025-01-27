# Fine-tune GPT model and create HealthBot application UI using Streamlit

# ==================== GET FINE-TUNED MODEL TO DEPLOY IN APP =========================
from fine_tuning_functions import createOpenAIClient, getFineTunedModelID

# create OpenAI client
client = createOpenAIClient()

fine_tuned_model_id = getFineTunedModelID(client)

# ========================== STREAMLIT AND CHAT SETUP =============================
import streamlit as st

# set up application UI
st.set_page_config(page_title="HealthBot: Your Virtual Doctor", layout="wide")

st.title("🩺🤖 HealthBot: Your Virtual Doctor")

# ========================== OPENAI API KEY HANDLING =============================
# accept OpenAI API key in a sidebar
with st.sidebar:
    st.markdown("**Enter your OpenAI API Key below:**")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("If you do not have an OpenAI API Key, you can get a key from [OpenAI](https://platform.openai.com/account/api-keys).")

# ================================== CHAT UI ======================================
st.header("How to use HealthBot:")
st.text("Enter in your symptoms separated by a comma in the text box below. "
        "HealthBot will try to find the most likely diagnosis based on the symptoms entered.")

avatars = {"human": "user", "ai": "assistant"}

# ask the user for symptoms
st.chat_message("assistant").write("What symptoms are you experiencing?")

# set up prompt textbox for user to enter symptoms
if user_symptoms := st.chat_input(placeholder="Enter your symptoms separated by a comma (e.g., fever, headaches, skin rash)"):
    st.chat_message("user").write(user_symptoms) # transfer prompt (which has the list of symptoms) to LLM

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # set the chatbot's role and instruction
    conversation_array = [{"role": "system", "content": "You are a medical professional. Try to give the most likely diagnosis based on given symptoms."}]

    # send user input and instruction to ChatGPT API as a prompt for answer
    # if user input is invalid, then return message asking user for a valid input
    try:
        conversation_array.append({"role": "system", "content": "What disease could I have if I am experiencing " + user_symptoms + " symptoms?"})

        # set up chatbot to use fine-tuned LLM to generate a response
        response = client.chat.completions.create(model=fine_tuned_model_id,
                                                  messages=conversation_array,
                                                  temperature=0)

        # retrieve, append to chat history, and display the chatbot response
        bot_response = response.choices[0].message.content

        conversation_array.append({"role": "assistant", "content": bot_response})

        st.chat_message("assistant").write(bot_response)

    except:
        st.chat_message("assistant").write("Please enter a list of symptoms (e.g., fever, headaches, skin rash)")
