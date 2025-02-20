from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
from PIL import Image
import io

app = FastAPI()

# Global objects for the model and feature extractor
model_path = "Andrew-Finch/vit-base-rocks"  # Replace with your actual model path
num_labels = 10  # Update this based on your number of labels
id2label = {
    0: 'Andesite',
    1: 'Basalt',
    2: 'Chalk',
    3: 'Dolomite',
    4: 'Flint',
    5: 'Gneiss',
    6: 'Granite',
    7: 'Limestone',
    8: 'Sandstone',
    9: 'Slate'
}
label2id = {
    'Andesite': 0,
    'Basalt': 1,
    'Chalk': 2,
    'Dolomite': 3,
    'Flint': 4,
    'Gneiss': 5,
    'Granite': 6,
    'Limestone': 7,
    'Sandstone': 8,
    'Slate': 9
}

model = ViTForImageClassification.from_pretrained(
    model_path,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)



@app.post("/classify/")
async def classify_image(query_id: str = Form(...), image: UploadFile = File(...)):
    try:
        # Read the image file
        image_content = await image.read()
        image = Image.open(io.BytesIO(image_content))

        # Preprocess the image (assuming you have a feature_extractor)
        inputs = feature_extractor(images=image, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        # Get the predicted class (assuming you have an id2label mapping)
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = id2label[predicted_class_idx]

        return {"query_id": query_id, "classification": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/")
async def get_model_info():
    return {
        "model_path": model_path,
        "num_labels": num_labels,
        "id2label": id2label,
        "label2id": label2id
    }

@app.get("/health/")
async def health_check():
    return {"status": "healthy"}

@app.post("/update_model/")
async def update_model(model_path: str = Form(...)):
    global model, feature_extractor
    try:
        # Update the model and feature extractor
        model = ViTForImageClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
        return {"message": "Model updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))