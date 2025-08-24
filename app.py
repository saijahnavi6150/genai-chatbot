import gradio as gr
from transformers import pipeline

# Load a free model from Hugging Face (no API key needed)
chatbot = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")

def chat_with_model(message):
    result = chatbot(message, max_new_tokens=100, do_sample=True)
    generated = result[0]["generated_text"]
    # Remove user's message from start if repeated
    return generated[len(message):].strip()

# Gradio UI
gr.Interface(
    fn=chat_with_model,
    inputs=gr.Textbox(lines=2, placeholder="Ask anything..."),
    outputs=gr.Textbox(label="Response"),
    title="My Free GenAI Chatbot",
    description="Built with Hugging Face Zephyr model, no API key required!"
).launch()