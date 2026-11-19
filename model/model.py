# Define the model's architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet34, ResNet34_Weights

class RefinedUpBlock(nn.Module):
    """Bilinear upsampling followed by projection and two spatial convolutions."""
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.project = nn.Conv2d(input_channels, output_channels, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(output_channels + skip_channels, output_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1), nn.ReLU(),
        )
        nn.init.dirac_(self.refine[2].weight)
        nn.init.zeros_(self.refine[2].bias)

    def forward(self, x, skip):
        x = self.project(F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False))
        return self.refine(torch.cat([x, skip], dim=1))


class Model(nn.Module):
    def __init__(self, pretrained=True, architecture='v1'):
        super().__init__()
        if architecture not in {'v1', 'v2'}:
            raise ValueError(f'Unknown model architecture: {architecture}')
        self.architecture = architecture

        # ResNet-34 Encoder
        resnet_encoder = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)

        self.conv1 = resnet_encoder.conv1
        self.bn1 = resnet_encoder.bn1
        self.relu = resnet_encoder.relu
        self.maxpool = resnet_encoder.maxpool

        self.layer1 = resnet_encoder.layer1
        self.layer2 = resnet_encoder.layer2
        self.layer3 = resnet_encoder.layer3
        self.layer4 = resnet_encoder.layer4

        self.configure_encoder('frozen')

        if architecture == 'v2':
            self.up1 = RefinedUpBlock(512, 256, 256)
            self.up2 = RefinedUpBlock(256, 128, 128)
            self.up3 = RefinedUpBlock(128, 64, 64)
            self.up4 = RefinedUpBlock(64, 64, 64)
            self.final_up = nn.Conv2d(64, 32, 1)
            self.full_resolution_stem = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            )
            self.full_resolution_refine = nn.Sequential(
                nn.Conv2d(48, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1),
            )
            # Start as a residual of zero, then learn full-resolution corrections.
            nn.init.zeros_(self.full_resolution_refine[2].weight)
            nn.init.zeros_(self.full_resolution_refine[2].bias)
            self.output = nn.Conv2d(32, 1, 1)
        else:
            # U-Net Decoder
            # Upsamples the image from bottleneck
            self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.conv1_d = nn.Conv2d(512, 256, 3, padding=1)

            self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.conv2_d = nn.Conv2d(256, 128, 3, padding=1)

            self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.conv3_d = nn.Conv2d(128, 64, 3, padding=1)

            self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
            self.conv4_d = nn.Conv2d(128, 64, 3, padding=1)

            self.final_up = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.output = nn.Conv2d(32, 1, 1)

    @classmethod
    def from_config(cls, config, pretrained=False):
        return cls(pretrained=pretrained, architecture=config.get('architecture', 'v1'))

    def encoder(self, x):
        # ResNet encoding
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        skip1 = x

        x = self.maxpool(x)
        x = self.layer1(x)
        skip2 = x

        x = self.layer2(x)
        skip3 = x

        x = self.layer3(x)
        skip4 = x

        bottleneck = self.layer4(x)

        return bottleneck, skip1, skip2, skip3, skip4

    def decoder(self, x, skip, up, conv):
        x = up(x)

        # Maintains same image dimensions from downsample
        # to ensure skip connections are same shape
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate downsample encoding to upsample
        x = torch.cat([x, skip], dim=1)

        x = F.relu(conv(x))
        return x

    def forward(self, x):
        input_size = x.size()[2:]

        # Encoder
        bottleneck, skip1, skip2, skip3, skip4 = self.encoder(x)

        if self.architecture == 'v2':
            features = self.up1(bottleneck, skip4)
            features = self.up2(features, skip3)
            features = self.up3(features, skip2)
            features = self.up4(features, skip1)
            features = self.final_up(F.interpolate(features, size=input_size, mode='bilinear', align_corners=False))
            fine = self.full_resolution_stem(x)
            features = features + self.full_resolution_refine(torch.cat([features, fine], dim=1))
            return self.output(features)

        # Decoder (bottom-up, U-Net architecture with skip connections)
        x = self.decoder(bottleneck, skip4, self.up1, self.conv1_d)
        x = self.decoder(x, skip3, self.up2, self.conv2_d)
        x = self.decoder(x, skip2, self.up3, self.conv3_d)
        x = self.decoder(x, skip1, self.up4, self.conv4_d)

        x = self.final_up(x)
        x = self.output(x)

        # Ensure output matches the same spatial dimension of input
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x


    def configure_encoder(self, mode='frozen'):
        """Unfreeze only the chosen late blocks; keep encoder BatchNorm stats fixed."""
        if mode not in {'frozen', 'layer4', 'layer3_4'}:
            raise ValueError(f'Unknown encoder training mode: {mode}')
        for name in ('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'):
            block = getattr(self, name)
            trainable = name == 'layer4' and mode != 'frozen' or name == 'layer3' and mode == 'layer3_4'
            for parameter in block.parameters():
                parameter.requires_grad = trainable
            block.eval()
        self.encoder_training = mode

    def training_parameter_groups(self, decoder_lr, encoder_lr):
        encoder_names = ('conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.')
        encoder, decoder = [], []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                (encoder if name.startswith(encoder_names) else decoder).append(parameter)
        groups = [{'params': decoder, 'lr': decoder_lr, 'name': 'decoder'}]
        if encoder:
            groups.append({'params': encoder, 'lr': encoder_lr, 'name': 'encoder'})
        return groups

    def train(self, mode=True):
        super().train(mode)
        # Fixed encoder running statistics also stabilize small-batch fine-tuning.
        for layer in (self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4):
            layer.eval()
        return self


def transfer_checkpoint(model, checkpoint):
    """Initialize a v2 model from trained v1 weights without silently dropping keys."""
    source_architecture = checkpoint['config'].get('architecture', 'v1')
    if model.architecture == source_architecture:
        model.load_state_dict(checkpoint['model'])
        return {'mode':'exact', 'source_architecture':source_architecture}
    if source_architecture != 'v1' or model.architecture != 'v2':
        raise ValueError(f'Unsupported architecture transfer: {source_architecture} -> {model.architecture}')
    source, destination = checkpoint['model'], model.state_dict()
    encoder_prefixes = ('conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.')
    for name in destination:
        if name.startswith(encoder_prefixes):
            destination[name] = source[name].clone()
    for number in range(1, 5):
        prefix = f'up{number}'
        destination[f'{prefix}.project.weight'] = source[f'{prefix}.weight'].mean(dim=(2,3)).transpose(0,1).unsqueeze(-1).unsqueeze(-1)
        destination[f'{prefix}.project.bias'] = source[f'{prefix}.bias'].clone()
        for suffix in ('weight','bias'):
            destination[f'{prefix}.refine.0.{suffix}'] = source[f'conv{number}_d.{suffix}'].clone()
    destination['final_up.weight'] = source['final_up.weight'].mean(dim=(2,3)).transpose(0,1).unsqueeze(-1).unsqueeze(-1)
    destination['final_up.bias'] = source['final_up.bias'].clone()
    destination['output.weight'] = source['output.weight'].clone()
    destination['output.bias'] = source['output.bias'].clone()
    model.load_state_dict(destination)
    return {'mode':'v1_to_v2', 'source_architecture':'v1',
            'encoder':'all trained weights and BatchNorm statistics copied exactly',
            'decoder':'matching spatial/output convolutions copied; upsampling projections use mean legacy kernel',
            'new_layers':'second stage convolutions initialized to identity; full-resolution residual initialized to zero'}
