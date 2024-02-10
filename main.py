### main.py
import gradio as gr
import torch
import torch.nn.functional as F
from load_model import SkinLesionCNN
import load_model

shared = True

model = SkinLesionCNN()
model.load_state_dict(torch.load(f="./models/01_SkinCancerDetectionModel.pth"))


# Define a function to make predictions with your model
def classify_image(image):
    # Preprocess the image
    preprocess = load_model.create_transformer()
    
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)
    print(image_tensor, type(image_tensor))

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        print(output)
    predicted_class = torch.max(output.data, 1)[1].data.squeeze()
    print(output.data)
    print(torch.max(output.data, 1))
    print(F.softmax(output, dim=1)[:, 1])

    return f"Prédiction : {'Malin' if predicted_class.item() == 0 else 'Benin'}"

demo = gr.Interface(
    fn=classify_image,
    inputs="image", 
    outputs="label", 
    title="Analyse de grain de beauté",
    description="Cette application, grâce à l'IA, donne un premier avis sur le risque de mélanome que présente un grain de beauté. Il ne constitue en rien une expertise médicale. Si vous avez un doute, prenez rendez-vous avec un spécialiste.")
demo.launch(share=shared)