from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import requests
import numpy as np
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from pydantic import BaseModel

class CaptionRequest(BaseModel):
    url: str
# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for HTML and assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

print("Model and Processor loaded successfully")

# Function to generate caption
def generate_caption(image: Image.Image):
    # Preprocess the image
    inputs = processor(image, return_tensors="pt")
    # Generate caption
    out = model.generate(**inputs)
    # Decode the generated caption
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_location = f"static/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        caption = generate_caption(Image.open(file_location))
        
        return JSONResponse(content={"caption": caption, "fileUrl": f"/{file_location}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})



@app.post("/generate_caption_url/")
async def generate_caption_url(request: CaptionRequest):
    try:
        image = Image.open(requests.get(request.url, stream=True).raw).convert("RGB")
        # Preprocess the image
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        
        return JSONResponse(content={"caption": caption, "imageUrl": request.url})
    except Exception as e:
        print(f"Error processing the image from URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
async def get_html():
    return JSONResponse(content={"detail": "Please visit /static/index.html to use the application."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
