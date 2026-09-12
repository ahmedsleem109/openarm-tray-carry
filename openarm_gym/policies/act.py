# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An action-chunking transformer policy, in the style of ACT.

Chosen over a vision-language-action model because the training budget here is
a single 6 GB laptop GPU: SmolVLA's 450M parameters fit in memory but its
optimizer states do not, and LoRA would confound the force-conditioning
ablation with an adapter-capacity effect. This is ~30M parameters, trains in
minutes, and isolates the variable under test.

The conditional VAE is not decoration. The puck task is deliberately
multimodal -- pushing and pick-and-place are both correct from the same
observation -- and a plain L1 regressor averages the two into a motion that is
neither. The latent gives the decoder somewhere to put that choice.

The force-conditioning ablation is the ``use_torque`` flag: it adds or removes
the joint-torque channel from the proprioceptive input and changes nothing
else.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


def _sinusoidal(length: int, dim: int) -> Tensor:
    """Return standard sinusoidal position embeddings."""
    position = torch.arange(length).unsqueeze(1).float()
    scale = torch.exp(
        torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
    )
    out = torch.zeros(length, dim)
    out[:, 0::2] = torch.sin(position * scale)
    out[:, 1::2] = torch.cos(position * scale)
    return out


class _VisionTower(nn.Module):
    """A ResNet-18 trunk shared across cameras, emitting spatial tokens."""

    def __init__(self, hidden: int, pretrained: bool = True) -> None:
        """Build the trunk and the projection to model width."""
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.trunk = nn.Sequential(*list(backbone.children())[:-2])
        self.project = nn.Conv2d(512, hidden, kernel_size=1)

    def forward(self, images: Tensor) -> Tensor:
        """Map ``(B, C, 3, H, W)`` images to ``(B, C*h*w, hidden)`` tokens."""
        batch, cameras = images.shape[:2]
        flat = images.flatten(0, 1)
        features = self.project(self.trunk(flat))
        tokens = features.flatten(2).transpose(1, 2)
        return tokens.reshape(batch, cameras * tokens.shape[1], -1)


class ACTPolicy(nn.Module):
    """Predict a chunk of future actions from images and proprioception."""

    def __init__(
        self,
        *,
        state_dim: int = 16,
        action_dim: int = 16,
        n_cameras: int = 3,
        chunk_size: int = 32,
        hidden: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        latent_dim: int = 32,
        use_torque: bool = True,
        pretrained_vision: bool = True,
    ) -> None:
        """Build the policy.

        Args:
            state_dim: Width of one proprioceptive vector (joint positions).
            action_dim: Width of one action.
            n_cameras: How many camera views are fed in.
            chunk_size: How many future actions are predicted at once.
            hidden: Model width.
            n_heads: Attention heads.
            n_encoder_layers: Transformer encoder depth.
            n_decoder_layers: Transformer decoder depth.
            latent_dim: Width of the CVAE latent.
            use_torque: Whether joint torque joins the proprioceptive input.
                This is the ablation switch.
            pretrained_vision: Whether to start from ImageNet weights.

        """
        super().__init__()
        self.chunk_size = chunk_size
        self.use_torque = use_torque
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        proprio_dim = state_dim * (2 if use_torque else 1)
        self.proprio_dim = proprio_dim

        self.vision = _VisionTower(hidden, pretrained=pretrained_vision)
        self.proprio_embed = nn.Linear(proprio_dim, hidden)
        self.latent_embed = nn.Linear(latent_dim, hidden)

        # CVAE encoder: sees the ground-truth chunk at training time only.
        self.cvae_action_embed = nn.Linear(action_dim, hidden)
        self.cvae_proprio_embed = nn.Linear(proprio_dim, hidden)
        self.cvae_cls = nn.Parameter(torch.zeros(1, 1, hidden))
        cvae_layer = nn.TransformerEncoderLayer(
            hidden, n_heads, hidden * 4, batch_first=True, norm_first=True
        )
        self.cvae_encoder = nn.TransformerEncoder(cvae_layer, 2)
        self.to_latent = nn.Linear(hidden, latent_dim * 2)

        encoder_layer = nn.TransformerEncoderLayer(
            hidden, n_heads, hidden * 4, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            hidden, n_heads, hidden * 4, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_decoder_layers)

        self.query_embed = nn.Parameter(torch.randn(1, chunk_size, hidden) * 0.02)
        self.head = nn.Linear(hidden, action_dim)

        self.register_buffer(
            "cvae_pos", _sinusoidal(chunk_size + 2, hidden).unsqueeze(0)
        )

    # ------------------------------------------------------------- plumbing

    def proprio(self, state: Tensor, torque: Tensor) -> Tensor:
        """Assemble the proprioceptive vector for this ablation arm."""
        return torch.cat([state, torque], dim=-1) if self.use_torque else state

    def encode_latent(
        self, proprio: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the CVAE posterior ``(mu, logvar)`` for a ground-truth chunk."""
        batch = proprio.shape[0]
        tokens = torch.cat(
            [
                self.cvae_cls.expand(batch, -1, -1),
                self.cvae_proprio_embed(proprio).unsqueeze(1),
                self.cvae_action_embed(actions),
            ],
            dim=1,
        )
        tokens = tokens + self.cvae_pos[:, : tokens.shape[1]]
        encoded = self.cvae_encoder(tokens)[:, 0]
        mu, logvar = self.to_latent(encoded).chunk(2, dim=-1)
        return mu, logvar

    def forward(
        self,
        images: Tensor,
        state: Tensor,
        torque: Tensor,
        actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Predict an action chunk.

        Args:
            images: ``(B, cameras, 3, H, W)``, already normalised.
            state: ``(B, state_dim)`` joint positions.
            torque: ``(B, state_dim)`` joint torques.
            actions: ``(B, chunk, action_dim)`` ground truth, training only.

        Returns:
            ``(predicted_chunk, mu, logvar)``. ``mu`` and ``logvar`` are
            ``None`` at inference, where the latent is set to its prior mean.

        """
        proprio = self.proprio(state, torque)
        batch = proprio.shape[0]

        if actions is not None:
            mu, logvar = self.encode_latent(proprio, actions)
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            mu = logvar = None
            latent = torch.zeros(
                batch, self.latent_dim, device=proprio.device, dtype=proprio.dtype
            )

        memory = torch.cat(
            [
                self.vision(images),
                self.proprio_embed(proprio).unsqueeze(1),
                self.latent_embed(latent).unsqueeze(1),
            ],
            dim=1,
        )
        memory = self.encoder(memory)

        queries = self.query_embed.expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        return self.head(decoded), mu, logvar


def act_loss(
    predicted: Tensor,
    target: Tensor,
    mu: Tensor | None,
    logvar: Tensor | None,
    kl_weight: float = 10.0,
) -> tuple[Tensor, dict[str, float]]:
    """Return the ACT training loss: L1 on the chunk plus a KL term."""
    l1 = F.l1_loss(predicted, target)
    if mu is None or logvar is None:
        return l1, {"l1": l1.detach().item(), "kl": 0.0}

    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)).mean()
    total = l1 + kl_weight * kl
    return total, {"l1": l1.detach().item(), "kl": kl.detach().item()}
