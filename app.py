import gradio as gr
import torch
import torch.nn as nn
from cnn_class import CNN

model = CNN()
model.load_state_dict(torch.load("CNN_model.pt"))
model.eval()

# Function to predict the digit sketch
def predict_digit_sketch(image):
    # Preprocess the image
    x = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.

    # Perform the prediction
    with torch.no_grad():
        output = model(x)
        _, predicted = torch.max(output, 1)

    # Return the predicted digit
    return str(predicted.item())

sp = gr.Sketchpad(shape=(28, 28))

demo = gr.Interface(
    fn=predict_digit_sketch,
    inputs=sp,
    outputs="text",
    title="Handwritten Digit Classifier",
    description="Draw a digit sketch and let the model predict the digit.",
)

demo.launch(share=True)
