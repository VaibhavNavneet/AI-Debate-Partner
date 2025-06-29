# unzip_model.py
import zipfile
import os

def unzip_model(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Model extracted to: {extract_to}")

if __name__ == "__main__":
    # Update these paths to match your files
    model_zip = "my_model.zip"
    tokenizer_zip = "tokenizer.zip"
    
    unzip_model(model_zip, "unzipped_model")
    unzip_model(tokenizer_zip, "unzipped_tokenizer")
# debate_app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

# Function to load model with caching
@st.cache_resource
def load_model():
    # Load configuration
    config = PeftConfig.from_pretrained("unzipped_model")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        device_map="auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unzipped_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, "unzipped_model")
    model.eval()
    
    return model, tokenizer

# Argument generation function
def generate_argument(topic: str, stance: str):
    if not st.session_state.model or not st.session_state.tokenizer:
        return "Model not loaded. Please load the model first."
    
    stance = stance.lower()
    if stance not in ["pro", "con"]:
        return "Stance must be 'pro' or 'con'"
    
    prompt = f"Topic: {topic}; Stance: {stance}; Argument:"
    inputs = st.session_state.tokenizer(
        prompt, 
        return_tensors="pt",
        max_length=256,
        truncation=True
    ).to(st.session_state.model.device)
    
    outputs = st.session_state.model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=st.session_state.tokenizer.eos_token_id,
        eos_token_id=st.session_state.tokenizer.eos_token_id
    )
    
    generated_text = st.session_state.tokenizer.decode(
        outputs[0], 
        skip_special_tokens=True
    )
    
    # Extract argument after the prompt
    if "Argument:" in generated_text:
        return generated_text.split("Argument:")[1].strip()
    return generated_text

# Streamlit UI
st.title("AI Debate Partner")
st.markdown("Generate arguments using your fine-tuned model")

# Load model button
if st.button("Load Model"):
    with st.spinner("Loading model..."):
        st.session_state.model, st.session_state.tokenizer = load_model()
    st.success("Model loaded successfully!")

# Input section
topic = st.text_input("Debate Topic", placeholder="e.g., 'School uniforms should be mandatory'")
stance = st.radio("Your Stance", ("Pro", "Con"))

# Generate button
if st.button("Generate Argument"):
    if not topic:
        st.warning("Please enter a debate topic")
    elif not st.session_state.model:
        st.error("Please load the model first")
    else:
        with st.spinner("Generating argument..."):
            argument = generate_argument(topic, stance)
            st.subheader("Generated Argument")
            st.write(argument)

# Model status
if st.session_state.model:
    st.info("Model is loaded and ready")
else:
    st.warning("Model not loaded yet")
