from tqdm.auto import trange
from einops import rearrange
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchviz import make_dot

from vector_quantize_pytorch import VectorQuantize


lr = 3e-4
train_iter = 1000
num_codes = 256
seed = 1234
device = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, **vq_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                VectorQuantize(dim=32, **vq_kwargs),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            ]
        )
        return

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, VectorQuantize):
                # x_shape = x.shape[:-1]
                image_shape = x.shape[-2:]
                # Following reshape is not right, collects not through channels (columns),
                # but pixels of one channel (one image) (visit Jupyter)
                # x_flat = x.view(x.size(0), -1, x.size(1))
                x_flat = rearrange(x, 'b c h w -> b (h w) c')
                x_flat, indices, commit_loss = layer(x_flat)
                # Need to transform back, once more not using view!
                # x = x_flat.contiguous().view(*x_shape, -1)
                x = rearrange(x_flat, 'b (h w) c -> b c h w', h=image_shape[0], w=image_shape[1])
            else:
                x = layer(x)
        return x.clamp(-1, 1), indices, commit_loss


def train(model, train_loader, train_iterations=1000, alpha=10):
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.to(device), y.to(device)

    for _ in (pbar := trange(train_iterations)):
        opt.zero_grad()
        x, _ = next(iterate_dataset(train_loader))
        out, indices, cmt_loss = model(x)
        rec_loss = (out - x).abs().mean()
        loss_summ = (rec_loss + alpha * cmt_loss)
        loss_summ.backward(retain_graph=True)
        """
        make_dot(loss_summ, params=dict(model.named_parameters()),
                 show_attrs=True,
                 show_saved=True).render("attached", format="svg")
        """
        opt.step()
        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            + f"cmt loss: {cmt_loss.item():.3f} | "
            + f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
        )
    return


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = DataLoader(
    datasets.FashionMNIST(
        root="~/data/fashion_mnist", train=True, download=True, transform=transform
    ),
    batch_size=256,
    shuffle=True,
)

torch.random.manual_seed(seed)
model = SimpleVQAutoEncoder(codebook_size=num_codes).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)
train(model, train_dataset, train_iterations=train_iter)
