import torch
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI
from pydantic import BaseModel
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
        imgs = generator(z).cpu().numpy()

    os.makedirs("outputs", exist_ok=True)

    paths = []
    for i in range(n):
        img = imgs[i][0]
        path = f"outputs/generated_{i+1}.png"
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()
        paths.append(os.path.abspath(path))
    
    return {"saved_images": paths}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
