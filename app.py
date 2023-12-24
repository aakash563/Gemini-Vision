import google.generativeai as genai
from PIL import Image
import gradio as gr
import numpy as np
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Now you can use hugging_face_api_key in your code

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro-vision')
def process_image_and_text(image, text):
  # Assuming image is the input from Gradio
  if text:
    image_array = np.asarray(image.data)  # Convert memoryview to NumPy array
    image = Image.fromarray(image_array.astype('uint8'), 'RGB')  # Now you can use astype
    response = model.generate_content([text, image])
    return response.text
  else:
    image_array = np.asarray(image.data)  # Convert memoryview to NumPy array
    image = Image.fromarray(image_array.astype('uint8'), 'RGB')  # Now you can use astype
    response = model.generate_content(["Tell me about this image in bulletin format", image])
    return response.text


iface = gr.Interface(
    process_image_and_text,
    inputs=["image", "textbox"],  # Specify image and text inputs
    outputs="textbox",          # Specify text output
    title="Image and Text Processor",  # Set the app title
)

iface.launch(debug=True, share=True)  # Launch the Gradio app
