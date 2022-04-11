import torch
import numpy as np
import torch.nn as nn


class SDE:

    def __init__(self, sigma=50):
        self.sigma = sigma

    def marginal_prob_std(self, t):
        return torch.sqrt((self.sigma ** (2 * t) - 1.) / 2. / np.log(self.sigma))

    def diffusion_coeff(self, t):
        return self.sigma ** t


class GaussianFourierProjection(nn.Module):

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class DownSamplingBlock(nn.Module):

    def __init__(self, in_c, out_c, act, emb_size, mp_size=2, max_gn=32, fact_gn=4):
        super(DownSamplingBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, (3, 3), padding='same')
        self.bn1 = nn.GroupNorm(min(out_c // fact_gn, max_gn), out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, (3, 3), padding='same')
        self.t_dense1 = nn.Linear(emb_size, emb_size)
        self.t_dense2 = nn.Linear(emb_size, out_c)
        self.bn2 = nn.GroupNorm(min(out_c // fact_gn, max_gn), out_c)
        self.max_pool = nn.MaxPool2d((mp_size, mp_size))
        self.act = act

    def forward(self, x, t_emb):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)

        t_emb = self.t_dense1(t_emb)
        t_emb = self.act(t_emb)
        t_emb = self.t_dense2(t_emb)
        t_emb = torch.unsqueeze(t_emb, -1)
        t_emb = torch.unsqueeze(t_emb, -1)

        x += t_emb
        x = self.bn2(x)
        x = self.act(x)
        mp = self.max_pool(x)
        return mp, x


class UpSamplingBlock(nn.Module):

    def __init__(self, in_c, out_c, act, emb_size, up_size=2, max_gn=32, fact_gn=4):
        super(UpSamplingBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, (3, 3), padding='same')
        self.bn1 = nn.GroupNorm(min(out_c // fact_gn, max_gn), out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, (3, 3), padding='same')
        self.t_dense1 = nn.Linear(emb_size, emb_size)
        self.t_dense2 = nn.Linear(emb_size, out_c)
        self.bn2 = nn.GroupNorm(min(out_c // fact_gn, max_gn), out_c)
        self.up_sample = nn.Upsample(scale_factor=up_size)
        self.act = act

    def forward(self, x, x_skip, t_emb):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)

        t_emb = self.t_dense1(t_emb)
        t_emb = self.act(t_emb)
        t_emb = self.t_dense2(t_emb)
        t_emb = torch.unsqueeze(t_emb, -1)
        t_emb = torch.unsqueeze(t_emb, -1)

        x += t_emb
        x = self.bn2(x)
        x = self.act(x)
        x = self.up_sample(x)
        x = torch.cat([x, x_skip], dim=1)
        return x


class ScoreNet(nn.Module):

    def __init__(self, sde, embed_dim=512, max_gn=32, fact_gn=4):
        emb_size = embed_dim
        super(ScoreNet, self).__init__()
        self.sde = sde
        self.emb = GaussianFourierProjection(emb_size)
        self.act = nn.LeakyReLU()
        self.dsb1 = DownSamplingBlock(3, 32, self.act, emb_size, max_gn=max_gn, fact_gn=fact_gn)
        self.dsb2 = DownSamplingBlock(32, 64, self.act, emb_size, max_gn=max_gn, fact_gn=fact_gn)
        self.dsb3 = DownSamplingBlock(64, 128, self.act, emb_size, max_gn=max_gn, fact_gn=fact_gn)
        self.dsb4 = DownSamplingBlock(128, 256, self.act, emb_size, max_gn=max_gn, fact_gn=fact_gn)
        self.middle = nn.Sequential(*[
            nn.Conv2d(256, 512, (2, 2)),
            nn.GroupNorm(min(max_gn, 512//fact_gn), 512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        ])
        self.usb4 = UpSamplingBlock(512, 256, self.act, emb_size, max_gn=max_gn, fact_gn=fact_gn)
        self.usb3 = UpSamplingBlock(512, 128, self.act, emb_size, max_gn=max_gn, fact_gn=fact_gn)
        self.usb2 = UpSamplingBlock(256, 64, self.act, emb_size, max_gn=max_gn, fact_gn=fact_gn)
        self.usb1 = UpSamplingBlock(128, 32, self.act, emb_size, max_gn=max_gn, fact_gn=fact_gn)
        self.finish = nn.Conv2d(64, 3, (1, 1))

    def encode(self, x, t_emb):
        x, h1 = self.dsb1(x, t_emb)
        x, h2 = self.dsb2(x, t_emb)
        x, h3 = self.dsb3(x, t_emb)
        x, h4 = self.dsb4(x, t_emb)
        return x, h1, h2, h3, h4

    def decode(self, x, h1, h2, h3, h4, t_emb):
        x = self.usb4(x, h4, t_emb)
        x = self.usb3(x, h3, t_emb)
        x = self.usb2(x, h2, t_emb)
        x = self.usb1(x, h1, t_emb)
        return x

    def forward(self, x, t):
        t_emb = self.emb(t)
        x, h1, h2, h3, h4 = self.encode(x, t_emb)
        x = self.middle(x)
        x = self.decode(x, h1, h2, h3, h4, t_emb)
        x = self.finish(x)
        x = x / self.sde.marginal_prob_std(t)[:, None, None, None]
        return x

    def loss(self, x, eps=1e-5):
        random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
        z = torch.randn_like(x)
        std = self.sde.marginal_prob_std(random_t)
        perturbed_x = x + z * std[:, None, None, None]
        score = self(perturbed_x, random_t)
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3)))
        return loss


class PredictorCorrector:

    def __init__(self, score_model, corrector_steps=2,
                 num_steps=1000, snr=0.1, device='cuda', eps=1e-4, shape=(3, 32, 32)):

        self.score_model = score_model.to(device)
        self.sde = score_model.sde
        self.num_steps = num_steps
        self.snr = snr
        self.device = device
        self.eps = eps
        self.corrector_steps = corrector_steps
        self.shape = shape
        self.time_steps = np.linspace(1., self.eps, self.num_steps)
        self.step_size = self.time_steps[0] - self.time_steps[1]

    def predictor_step(self, x, t):
        g = self.sde.diffusion_coeff(t)
        x_mean = x + (g ** 2)[:, None, None, None] * self.score_model(x, t) * self.step_size
        x = x_mean + torch.sqrt(g ** 2 * self.step_size)[:, None, None, None] * torch.randn_like(x)
        return x, x_mean

    def corrector_step(self, x, t):
        grad = self.score_model(x, t)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = np.sqrt(np.prod(x.shape[1:]))
        langevin_step_size = 2 * (self.snr * noise_norm / grad_norm) ** 2
        x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
        return x

    def sample(self, batch_size):
        init_x = torch.randn(batch_size,
                             self.shape[0], self.shape[1], self.shape[2], device=self.device)
        init_x *= self.sde.marginal_prob_std(torch.ones(batch_size, device=self.device))[:, None, None, None]
        x = init_x

        self.score_model.eval()
        with torch.no_grad():
            for time_step in self.time_steps:
                batch_time_step = torch.ones(batch_size, device=self.device)
                batch_time_step *= time_step
                # Corrector step (Langevin MCMC)
                for _ in range(self.corrector_steps):
                    x = self.corrector_step(x, batch_time_step)

                # Predictor step (Euler-Maruyama)
                x, x_mean = self.predictor_step(x, batch_time_step)
        self.score_model.train()

        return x_mean


class ExponentialMovingAverage:

    def __init__(self, parameters, decay):
        self.decay = decay
        self.num_updates = 0
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        decay = self.decay
        self.num_updates += 1
        decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
