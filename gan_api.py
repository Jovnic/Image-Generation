from fastapi import FastAPI
from pydantic import BaseModel
import torch
from helper_lib.model import Generator

app = FastAPI(title="GAN Image Generator API")

class InputData(BaseModel):
    n_images: int = 1 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator()
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.to(device)
generator.eval()

@app.post("/generate")
def generate_images(data: InputData):
    n = data.n_images
    z = torch.randn(n, 100, device=device)
    with torch.no_grad():
        imgs = generator(z).cpu().numpy().tolist()
    return {"generated_images": imgs}
