"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams.update(hparams)
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        from torchvision import models

        self.pretrained = models.mobilenet_v2(pretrained=True).features
        for para in self.pretrained.parameters():
            para.requires_grad = False
        
        self.middle_conv = nn.Conv2d(1280,320,1,1)
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(320,240,4,2,1),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(240,160,4,2,2),
            nn.ReLU(),
            nn.Dropout()
        )
        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose2d(160,80,4,2,1),
            nn.LeakyReLU(),
            nn.Dropout()
        )
        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose2d(80,40,4,2,1),
            nn.ReLU(),
            nn.Dropout()
        )
        self.up_conv5 = nn.Sequential(
            nn.ConvTranspose2d(40,num_classes,4,2,1),
        )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.pretrained(x)
        x = self.middle_conv(x)
        x = self.up_conv1(x)
        x = self.up_conv2(x)
        x = self.up_conv3(x)
        x = self.up_conv4(x)
        x = self.up_conv5(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
