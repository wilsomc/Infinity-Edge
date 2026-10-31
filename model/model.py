# Define the model's architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torchvision.models import resnet34, ResNet34_Weights

class Model(nn.Module):
    trainloader = None
    testloader = None

    MASKED_L1_WEIGHT = 0.8
    GRAD_LOSS_WEIGHT = 0.2

    MAX_EPOCH = 1
    LEARNING_RATE = 1e-3

    def __init__(self, trainloader, testloader):
        nn.Module.__init__(self)
        self.trainloader = trainloader
        self.testloader = testloader

        # ResNet-34 Encoder
        resnet_encoder = resnet34(weights=ResNet34_Weights.DEFAULT)

        self.conv1 = resnet_encoder.conv1
        self.bn1 = resnet_encoder.bn1
        self.relu = resnet_encoder.relu
        self.maxpool = resnet_encoder.maxpool

        self.layer1 = resnet_encoder.layer1
        self.layer2 = resnet_encoder.layer2
        self.layer3 = resnet_encoder.layer3
        self.layer4 = resnet_encoder.layer4

        # Freeze encoder parameters
        # TODO: remove if doesn't work
        encoder_layers = [
            self.conv1,
            self.bn1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        ]
        for layer in encoder_layers:
            for parameter in layer.parameters():
                parameter.requires_grad = False
            layer.eval()

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

    # Masked SmoothL1Loss, returns loss and total accounted pixels in a batch
    def masked_smooth_l1(self, criterion, prediction, target, mask):
        valid_pixel = mask.bool()
        valid_pixel_count = int(valid_pixel.sum().item())

        if valid_pixel_count == 0:
            return None, 0

        # replace invalid values with prefilled value
        prediction = prediction.masked_fill(~valid_pixel, 0)
        target = target.masked_fill(~valid_pixel, 0)

        print("Prediction shape:", prediction.shape)
        print("Depth shape:", target.shape)
        print("Mask shape:", mask.shape)
        print("Valid pixels:", valid_pixel_count)

        loss = criterion(prediction, target)

        loss = (loss * valid_pixel).sum() / valid_pixel_count
        return loss, valid_pixel_count


    def forward(self, x):
        input_size = x.size()[2:]

        # Encoder
        bottleneck, skip1, skip2, skip3, skip4 = self.encoder(x)

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


    def fit(self, trainloader, device):
        smooth_loss = nn.SmoothL1Loss(reduction='none')
        optimizer = optim.Adam(self.parameters(), lr=self.LEARNING_RATE)

        loss_history = []


        for epoch in range(self.MAX_EPOCH):

            total_loss = 0.0
            total_valid_pixels = 0

            running_loss = 0.0
            running_valid_pixels = 0

            for i, (rgb, depth, mask) in enumerate(trainloader):

                rgb_batch = rgb.to(device)
                depth_batch = depth.to(device)
                mask_batch = mask.to(device)

                optimizer.zero_grad()
                outputs = self(rgb_batch)

                # calculate average loss of all pixels in batch
                # loss function: aL1_loss + bDepthGrad_Loss
                l1_loss, valid_pixel_count = self.masked_smooth_l1(smooth_loss, outputs, depth_batch, mask_batch)


                if l1_loss is None:
                    continue

                # loss = (self.MASKED_L1_WEIGHT * l1_loss)
                loss = l1_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * valid_pixel_count
                total_valid_pixels += valid_pixel_count

                running_loss += loss.item() * valid_pixel_count
                running_valid_pixels += valid_pixel_count

                # print statistics
                if i % 2 == 0:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / running_valid_pixels))
                    running_loss = 0.0
                    running_valid_pixels = 0

            epoch_loss = total_loss / total_valid_pixels
            loss_history.append(epoch_loss)

        print("finished training")

    def run(self, device):
        print('method running...')
        print('--start training...')
        self.to(device)
        self.fit(self.trainloader, device)
        print('--start testing...')
        # self.test(self.testloader)
        return {'Training / Testing Finished'}
