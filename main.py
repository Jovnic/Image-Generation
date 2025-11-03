import torch
from torch import optim
from helper_lib.data_loader import get_data_loaders
from helper_lib.model import SimpleCNN
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.utils import save_model, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_data_loaders(batch_size=64)
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    loss = train_model(model, train_loader, optimizer, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

test_loss, acc = evaluate_model(model, test_loader, device)
print(f"Test Loss: {test_loss:.4f}, Accuracy: {acc:.2f}%")

save_model(model, "cnn.pth")
print("Model saved as cnn.pth")

model = load_model(model, "cnn.pth", device)
print("Model reloaded successfully.")

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn

from helper_lib.model import Generator, Discriminator
from helper_lib.trainer import train_gan

latent_dim = 100
batch_size = 64
epochs = 20
lr = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = Generator(latent_dim=latent_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

train_gan(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, epochs, device)

