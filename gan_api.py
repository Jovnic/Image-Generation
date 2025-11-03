import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import io
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from helper_lib.model import Generator

app = FastAPI(title="GAN Image Generator API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)

if os.path.exists("generator.pth"):
    generator.load_state_dict(torch.load("generator.pth", map_location=device))

generator.eval()

@app.get("/gan/generate_png")
def generate_png(num: int = 16, nrow: int = 4):
    z = torch.randn(num, 100, device=device)
    with torch.no_grad():
        imgs = generator(z).cpu()

    grid = make_grid(imgs, nrow=nrow, normalize=True)
    plt.figure(figsize=(nrow, nrow))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
