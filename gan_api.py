from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os
from helper_lib.model import Generator

app = FastAPI(title="GAN Image Generator API", version="1.0")

class InputData(BaseModel):
    n_images: int = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)

if os.path.exists("generator.pth"):
    generator.load_state_dict(torch.load("generator.pth", map_location=device))

generator.eval()

@app.post("/generate")
def generate_images(data: InputData):
    n = data.n_images
    z = torch.randn(n, 100, device=device)
    with torch.no_grad():
        imgs = generator(z).cpu().numpy().tolist()
    return {"generated_images": imgs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
