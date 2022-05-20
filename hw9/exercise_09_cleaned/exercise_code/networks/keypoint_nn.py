"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super(KeypointModel, self).__init__()
        self.hparams.update(hparams)
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################

        output_size_checker = torch.zeros(*hparams['input_shape'])
        
        self.conv2d = [
            nn.Sequential(
                nn.Conv2d(i, o, kernel_size=3, padding=1),
                nn.LeakyReLU(0.3),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            for i, o in zip(hparams['n_input_channel'], hparams['n_output_channel'])
        ]
        for i in range(len(self.conv2d)):
            setattr(self, 'conv2d_'+str(i), self.conv2d[i])
        
        self.flatten = nn.Flatten()
        
        # ====== check flatten layer output size (just lazy to calculate) =====
        for c in self.conv2d:
            output_size_checker = c(output_size_checker)
        output_size_checker = self.flatten(output_size_checker)

        flatten_output_size = output_size_checker.shape[1]
        # =============================================
        
        num_hidden_layers = len(hparams['n_hidden'])
        self.dense = [
            nn.Sequential(
                nn.Linear(flatten_output_size, hparams['n_hidden'][0]),
                nn.LeakyReLU()
            )
        ]
        if num_hidden_layers > 1:
            self.dense += [
                nn.Sequential(
                    nn.Linear(hparams['n_hidden'][i-1], hparams['n_hidden'][i]),
                    nn.LeakyReLU()
                )
                for i in range(1, num_hidden_layers)
            ]
        for i in range(len(self.dense)):
            setattr(self, 'dense_'+str(i), self.dense[i])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################

        for c in self.conv2d:
            x = c(x)
        x = self.flatten(x)
        for d in self.dense:
            x = d(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
