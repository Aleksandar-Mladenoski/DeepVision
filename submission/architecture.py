import torch.nn as nn
import torch

class MyCNN(torch.nn.Module):
    def __init__(
            self,
            input_channels: int,
            hidden_channels: list,
            fnn_channels: list,
            use_batchnormalization: bool,
            num_classes: int,
            kernel_size: list,
            dropout_rate_fnn,
            dropout_rate_cnn,
            activation_function: torch.nn.Module = torch.nn.ReLU()
    ):
        """
        :param input_channels: specifies the number of channels (e.g. color-channels) in the input to the neural network.
        :param hidden_channels: specifies the number of feature channels that are computed by the
                convolutional layers. The input is a list, where the first, second and third elements specify
                the number of output features in the first, second, and third hidden layers, respectively.
        :param use_batchnormalization: controls whether 2-dimensional batch normalization is used
                after every convolutional layer.
        :param num_classes: specifies the number of output neurons of the fully-connected output layer
                of the neural network.
        :param kernel_size: specifies the size of kernels in each hidden layer of the neural network. The
                input is a list, where the first, second and third elements specify the size of the kernel in
                the first, second, and third hidden layers, respectively.
        :param activation_function: specifies which non-linear activation function is used after all non-
                output layers of the network. By default, the network should use the torch.nn.ReLU
                (module) function.
        """
        super().__init__()
        old_out = input_channels
        self.hidden_conv = torch.nn.ModuleList()
        self.kernel_sizes = kernel_size
        self.pool = torch.nn.MaxPool2d(2, 2)  # Max pooling to reduce the spatial dimensions

        current_width, current_height = 100, 100  # Starting from 100x100 input images

        for i in range(len(hidden_channels)):
            if use_batchnormalization: 
                if i % 2 == 0 and i != 0:
                    layer = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=old_out, out_channels=hidden_channels[i], kernel_size=kernel_size[i],
                                        padding='same', padding_mode='zeros'),
                        torch.nn.BatchNorm2d(hidden_channels[i]),
                        self.pool,
                        torch.nn.Dropout2d(dropout_rate_cnn)
                    )
                else:
                    layer = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=old_out, out_channels=hidden_channels[i], kernel_size=kernel_size[i],
                                        padding='same', padding_mode='zeros'),
                        torch.nn.BatchNorm2d(hidden_channels[i]),
                        torch.nn.Dropout2d(dropout_rate_cnn)
                    )
            else:
                layer = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=old_out, out_channels=hidden_channels[i],
                                    kernel_size=kernel_size[i], padding='same', padding_mode='zeros'),
                    self.pool,
                    torch.nn.Dropout2d(dropout_rate_cnn)
                )
            self.hidden_conv.append(layer)
            old_out = hidden_channels[i]

            # Update the current dimensions after convolution and pooling
            #current_width = math.floor((current_width + 2 * 0 - (kernel_size[i] - 1) - 1) / 2 + 1)
            #current_height = math.floor((current_height + 2 * 0 - (kernel_size[i] - 1) - 1) / 2 + 1)

        # Calculate the input size for the first fully connected layer
        self.conv_output_size = 4608

        # Modify the FNN setup
        self.fnn = torch.nn.ModuleList()
        in_features = self.conv_output_size
        for out_features in fnn_channels:
            if use_batchnormalization:
                layer = torch.nn.Sequential(
                    torch.nn.Linear(in_features, out_features),
                    torch.nn.BatchNorm1d(out_features),
                                        torch.nn.Dropout(dropout_rate_fnn)
                )
            else:
                layer = torch.nn.Sequential(
                    torch.nn.Linear(in_features, out_features),
                    torch.nn.Dropout(dropout_rate_fnn))
            self.fnn.append(layer)
            in_features = out_features

        self.output_layer = torch.nn.Linear(fnn_channels[-1], num_classes)
        self.ac = activation_function
        self.flatten = torch.nn.Flatten()

    def forward(self, input_images):
        for layer in self.hidden_conv:
            input_images = layer(input_images)
            input_images = self.ac(input_images)

        input_images = self.flatten(input_images)

        for fnn_layer in self.fnn:
            input_images = fnn_layer(input_images)
            input_images = self.ac(input_images)

        input_images = self.output_layer(input_images)

        return input_images  
    


model = MyCNN(
        input_channels=1, 
        hidden_channels=[32, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512], 
        fnn_channels=[2048, 512, 128, 64],
        num_classes=20,
        dropout_rate_fnn=0.4,
        dropout_rate_cnn=0.20,
        kernel_size=[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        use_batchnormalization=True
    ) 


