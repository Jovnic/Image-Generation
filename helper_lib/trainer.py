import torch
import torch.nn.functional as F

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

import torch
import torch.nn as nn

def train_gan(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, epochs, device):
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        g_loss_total, d_loss_total = 0.0, 0.0

        for imgs, _ in dataloader:
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)

            valid = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)

            optimizer_G.zero_grad()

            z = torch.randn(batch_size, 100, device=device)
            gen_imgs = generator(z)

            g_loss = criterion(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            real_loss = criterion(discriminator(real_imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] | D loss: {d_loss_total/len(dataloader):.4f} | G loss: {g_loss_total/len(dataloader):.4f}")

    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
