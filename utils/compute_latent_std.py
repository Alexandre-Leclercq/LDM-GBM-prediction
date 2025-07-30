import torch
from tqdm import tqdm


def compute_latent_std(model, train_loader, idx_data):
    latents = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(train_loader, position=0, leave=True, desc='Encoding'):
            # we select our data as our dataset contain multiple modalities
            batch = batch[idx_data].to(device)
            z_batch, _, _ = model.encode(batch)
            latents.append(z_batch.detach())
    latents = torch.cat(latents, dim=0)

    return latents.std()
