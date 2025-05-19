import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import skew, kurtosis

# ❗️FOR RTX A5000❗️
torch.set_float32_matmul_precision('high')

# --------------------------- Models -----------------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
        )
        self.tail = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, return_features=False):
        features = self.feat(x)
        out = self.tail(features)
        return (out, features) if return_features else out

# --------------------- Gradient Penalty -------------------------------------
def gradient_penalty(critic, real, fake, device, lambda_gp):
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, device=device).expand_as(real)
    interpolates = eps * real + (1 - eps) * fake
    interpolates.requires_grad_(True)
    d_interpolates = critic(interpolates)
    grads = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True, retain_graph=True
    )[0]
    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)
    gp = ((grad_norm - 1) ** 2).mean()
    return lambda_gp * gp, grad_norm.mean().item()

# --------------------------- Train WGAN-GP ----------------------------------
def train(args):
    df = pd.read_csv(args.input_csv)
    df = df.drop(columns=['status_label', 'year', 'year_n'], errors='ignore')
    feat_cols = [f'X{i}' for i in range(1, 19)]
    data = np.hstack([df['status'].values.reshape(-1, 1), df[feat_cols].values]).astype(np.float32)

    dataset = TensorDataset(torch.tensor(data))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                        num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    os.makedirs("generated", exist_ok=True)
    log_path = "logs.txt"
    with open(log_path, "w") as logf:
        logf.write("epoch,lossD,lossG,grad_norm_D,grad_norm_G,gp_norm\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = data.shape[1]
    G = Generator(args.noise_dim, dim, args.hidden_dim).to(device)
    D = Critic(dim, args.hidden_dim).to(device)
    optG = optim.Adam(G.parameters(), lr=args.lrG, betas=args.betaG, weight_decay=1e-4)
    optD = optim.Adam(D.parameters(), lr=args.lrD, betas=args.betaD)

    for epoch in range(1, args.epochs + 1):
        for real_batch, in loader:
            real = real_batch.to(device)
            bsz = real.size(0)
            for _ in range(args.critic_iters):
                noise = torch.randn(bsz, args.noise_dim, device=device)
                fake = G(noise).detach()
                lossD = D(fake).mean() - D(real).mean()
                gp, gp_norm = gradient_penalty(D, real, fake, device, args.lambda_gp)
                optD.zero_grad()
                total_lossD = lossD + gp
                total_lossD.backward()
                grad_norm_D = sum(p.grad.data.norm(2).item()**2 for p in D.parameters() if p.grad is not None)**0.5
                optD.step()

            noise = torch.randn(bsz, args.noise_dim, device=device)
            fake = G(noise)
            D_fake_out = D(fake)
            loss_adv = -D_fake_out.mean()
            lossG = loss_adv
            optG.zero_grad()
            lossG.backward()
            grad_norm_G = sum(p.grad.data.norm(2).item()**2 for p in G.parameters() if p.grad is not None)**0.5
            optG.step()

        with open(log_path, "a") as logf:
            logf.write(f"{epoch},{lossD.item():.4f},{lossG.item():.4f},{grad_norm_D:.4f},{grad_norm_G:.4f},{gp_norm:.4f}\n")

        if epoch % args.log_every == 0:
            print(f"Epoch {epoch}/{args.epochs} | D {lossD.item():.3f} | G {lossG.item():.3f}")
            with torch.no_grad():
                z = torch.randn(5000, args.noise_dim, device=device)
                synth = G(z).cpu().numpy()
                features = ['status'] + [f'X{i}' for i in range(1, 19)]
                synth_df = pd.DataFrame(synth, columns=features)
                real_df = pd.DataFrame(data, columns=features)

                fig, axes = plt.subplots(4, 5, figsize=(20, 12))
                axes = axes.flatten()
                # Write stats file for losses, grad norms and means/stds
                with open(f"generated/stats_epoch_{epoch}.txt", "w") as fstats:
                    fstats.write(
                        f"LossD: {lossD.item():.4f}, "
                        f"LossG: {lossG.item():.4f}, "
                        f"GradNormD: {grad_norm_D:.4f}, "
                        f"GradNormG: {grad_norm_G:.4f}, "
                        f"GP_Norm: {gp_norm:.4f}\n"
                    )
                    for feat in features:
                        real_vals = real_df[feat]
                        synth_vals = synth_df[feat]
                        fstats.write(
                            f"{feat} | "
                            f"Real: mean={real_vals.mean():.4f}, std={real_vals.std():.4f} | "
                            f"Generated: mean={synth_vals.mean():.4f}, std={synth_vals.std():.4f}\n"
                        )

                for i, feat in enumerate(features):
                    ax = axes[i]
                    bins = np.linspace(
                        min(real_df[feat].min(), synth_df[feat].min()),
                        max(real_df[feat].max(), synth_df[feat].max()), 100
                    )
                    ax.hist(real_df[feat], bins=bins, alpha=0.5, label='Real', density=True, color='orange')
                    ax.hist(synth_df[feat], bins=bins, alpha=0.5, label='Generated', density=True, color='blue')
                    ax.set_title(feat)
                    ax.tick_params(axis='x', rotation=45)

                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', ncol=2)
                fig.delaxes(axes[-1])
                plt.tight_layout()
                plt.savefig(f"generated/feature_compare_epoch_{epoch}.png")
                plt.close()

    torch.save(G.state_dict(), args.out)
    print(f"✅ Generator saved to {args.out}")

# ------------------------- Generation ---------------------------------------
def generate(args):
    with open(args.scalers, 'rb') as f:
        scs = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 1 + 18
    G = Generator(args.noise_dim, dim, args.hidden_dim).to(device)
    G.load_state_dict(torch.load(args.weights, map_location=device))
    G.eval()

    z = torch.randn(args.num_samples, args.noise_dim, device=device)
    with torch.no_grad():
        synth = G(z).cpu().numpy()

    status = (synth[:, 0] > 0).astype(int)
    feats_n = synth[:, 1:]

    df_out = pd.DataFrame()
    df_out['company_name'] = [f"C_syn{i+1}" for i in range(args.num_samples)]
    df_out['status_label'] = np.where(status, 'alive', 'failed')

    feat_cols = [f"X{i}" for i in range(1, 19)]
    if isinstance(scs, dict) and all(col in scs for col in feat_cols):
        feats_orig = np.zeros_like(feats_n)
        for i, col in enumerate(feat_cols):
            col_min = scs[col]['min']
            col_max = scs[col]['max']
            feats_orig[:, i] = ((feats_n[:, i] + 1) / 2) * (col_max - col_min) + col_min
    else:
        raise KeyError("No suitable scaler found in scalers.pkl")

    df_feats = pd.DataFrame(feats_orig, columns=feat_cols)
    df_out = pd.concat([df_out, df_feats], axis=1)
    df_out.to_csv(args.out_csv, index=False)
    print(f"📦 Generated {args.num_samples} synthetic profiles → {args.out_csv}")

# -------------------------- Main entrypoint ---------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    tr = sub.add_parser('train')
    tr.add_argument('--input_csv', default='data_normalized.csv')
    tr.add_argument('--scalers', default='scalers.pkl')
    tr.add_argument('--epochs', type=int, default=100)
    tr.add_argument('--batch_size', type=int, default=8192)
    tr.add_argument('--noise_dim', type=int, default=256)
    tr.add_argument('--hidden_dim', type=int, default=1024)
    tr.add_argument('--lrG', type=float, default=1e-5)
    tr.add_argument('--lrD', type=float, default=3e-5)
    tr.add_argument('--betaG', nargs=2, type=float, default=(0.5, 0.99))
    tr.add_argument('--betaD', nargs=2, type=float, default=(0.5, 0.99))
    tr.add_argument('--critic_iters', type=int, default=5)
    tr.add_argument('--lambda_gp', type=float, default=10)
    tr.add_argument('--log_every', type=int, default=10)
    tr.add_argument('--out', default='generator_wgangp.pth')

    gen = sub.add_parser('generate')
    gen.add_argument('--weights', default='generator_wgangp.pth')
    gen.add_argument('--scalers', default='scalers.pkl')
    gen.add_argument('--num_samples', type=int, default=50000)
    gen.add_argument('--noise_dim', type=int, default=256)
    gen.add_argument('--hidden_dim', type=int, default=1024)
    gen.add_argument('--out_csv', default='synthetic.csv')

    args = parser.parse_args()
    if args.cmd == 'train':
        train(args)
    elif args.cmd == 'generate':
        generate(args)
    else:
        parser.print_help()