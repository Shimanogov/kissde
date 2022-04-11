import numpy as np
import torch
import wandb
from torchvision.utils import make_grid

from model import SDE, PredictorCorrector, ExponentialMovingAverage
from model import ScoreNet
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


def main():
    wandb.init(project='score_sde')

    wandb.config.device = 'cuda'
    wandb.config.pc_sample_size = 16
    wandb.config.corrector_steps = 1
    wandb.config.sigma = 50
    wandb.config.embed_dim = 512
    wandb.config.pc_num_steps = 1000
    wandb.config.pc_snr = 0.15
    wandb.config.n_epochs = 100 * 1000
    wandb.config.batch_size = 2048
    wandb.config.lr = 1e-2
    wandb.config.data_loader_workers = 15
    wandb.config.eval_log_freq = 100
    wandb.config.save_model = 1000
    wandb.config.decay = 0.999
    wandb.config.lr_decay = 0.9999
    wandb.config.max_gn = 32
    wandb.config.fact_gn = 4
    wandb.config.clipnorm = 1.

    sde = SDE(sigma=wandb.config.sigma)
    model = ScoreNet(sde, embed_dim=wandb.config.embed_dim, max_gn=wandb.config.max_gn, fact_gn=wandb.config.fact_gn)
    pc = PredictorCorrector(model,
                            device=wandb.config.device, corrector_steps=wandb.config.corrector_steps,
                            snr=wandb.config.pc_snr, num_steps=wandb.config.pc_num_steps)

    dataset = CIFAR10('.', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=wandb.config.batch_size,
                             shuffle=True, num_workers=wandb.config.data_loader_workers)
    optimizer = Adam(pc.score_model.parameters(), lr=wandb.config.lr)
    exp_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, wandb.config.lr_decay)
    ema = ExponentialMovingAverage(pc.score_model.parameters(), wandb.config.decay)

    for epoch in range(wandb.config.n_epochs):
        wandb_dict = {}
        avg_loss = 0.
        num_items = 0
        for x, _ in data_loader:
            x = x.to(wandb.config.device)
            loss = pc.score_model.loss(x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pc.score_model.parameters(), wandb.config.clipnorm)
            optimizer.step()
            ema.update(pc.score_model.parameters())
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        if epoch % wandb.config.eval_log_freq == 0:
            samples = pc.sample(wandb.config.pc_sample_size)
            samples = samples.clamp(0.0, 1.0)
            sample_grid = make_grid(samples, nrow=int(np.sqrt(wandb.config.pc_sample_size)))
            sample_grid = wandb.Image(sample_grid)

            ema.store(pc.score_model.parameters())
            ema.copy_to(pc.score_model.parameters())

            e_samples = pc.sample(wandb.config.pc_sample_size)
            e_samples = e_samples.clamp(0.0, 1.0)
            e_sample_grid = make_grid(e_samples, nrow=int(np.sqrt(wandb.config.pc_sample_size)))
            e_sample_grid = wandb.Image(e_sample_grid)

            ema.restore(pc.score_model.parameters())

            wandb_dict['Samples'] = sample_grid
            wandb_dict['EMA Samples'] = e_sample_grid

        exp_lr.step()

        if epoch % wandb.config.save_model == 0:
            ema.store(pc.score_model.parameters())
            ema.copy_to(pc.score_model.parameters())
            torch.save(pc.score_model.state_dict(), 'ema_ckpt.pth')
            ema.restore(pc.score_model.parameters())
            torch.save(pc.score_model.state_dict(), 'ckpt.pth')

            wandb.log_artifact('ckpt.pth', 'model_state_dict', 'model_state_dict')
            wandb.log_artifact('ema_ckpt.pth', 'ema_model_state_dict', 'ema_model_state_dict')

        wandb_dict['Average loss'] = avg_loss / num_items
        wandb_dict['lr'] = exp_lr.get_last_lr()[0]

        wandb.log(wandb_dict)


if __name__ == '__main__':
    main()
