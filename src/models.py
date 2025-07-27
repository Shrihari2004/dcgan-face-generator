import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
    Optimized Generator network for DCGAN architecture.
    Reduced feature dimensions while maintaining quality through better architecture.
    """
    def __init__(self, latent_dim=100, ngf=48, channels=3):  # Reduced from 64 to 48
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.ngf = ngf
        self.channels = channels
        
        # Initial linear layer to project noise to feature maps
        self.linear = nn.Linear(latent_dim, ngf * 8 * 4 * 4)
        
        # Transposed convolution layers for upsampling
        # Each layer doubles the spatial dimensions
        self.conv_transpose1 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.conv_transpose2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.conv_transpose3 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.conv_transpose4 = nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False)
        
        # Batch normalization layers for stable training
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        self.bn2 = nn.BatchNorm2d(ngf * 4)
        self.bn3 = nn.BatchNorm2d(ngf * 2)
        self.bn4 = nn.BatchNorm2d(ngf)
        
        # Reduced dropout for better performance
        self.dropout = nn.Dropout(0.2)  # Reduced from 0.3
        
    def forward(self, x):
        """
        Forward pass through the generator.
        Args:
            x: Input noise tensor of shape (batch_size, latent_dim)
        Returns:
            Generated image tensor of shape (batch_size, channels, height, width)
        """
        # Linear projection and reshape to feature maps
        x = self.linear(x)
        x = x.view(x.size(0), self.ngf * 8, 4, 4)
        
        # Transposed convolution layers with batch norm and ReLU
        x = F.relu(self.bn1(x))
        x = self.conv_transpose1(x)
        x = F.relu(self.bn2(x))
        x = self.dropout(x)
        
        x = self.conv_transpose2(x)
        x = F.relu(self.bn3(x))
        x = self.dropout(x)
        
        x = self.conv_transpose3(x)
        x = F.relu(self.bn4(x))
        
        # Final layer with tanh activation for image generation
        x = self.conv_transpose4(x)
        x = torch.tanh(x)
        
        return x

class Discriminator(nn.Module):
    """
    Optimized Discriminator network for DCGAN architecture.
    Reduced feature dimensions while maintaining discriminative power.
    """
    def __init__(self, ndf=48, channels=3):  # Reduced from 64 to 48
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.channels = channels
        
        # Convolutional layers for feature extraction
        # Each layer halves the spatial dimensions
        self.conv1 = nn.Conv2d(channels, ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(ndf)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        
        # Reduced dropout for better performance
        self.dropout = nn.Dropout(0.2)  # Reduced from 0.3
        
        # Final linear layer for classification
        self.linear = nn.Linear(ndf * 8 * 4 * 4, 1)
        
    def forward(self, x):
        """
        Forward pass through the discriminator.
        Args:
            x: Input image tensor of shape (batch_size, channels, height, width)
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        # Convolutional layers with LeakyReLU and batch norm
        x = F.leaky_relu(self.conv1(x), 0.2)
        
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x), 0.2)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = F.leaky_relu(self.bn3(x), 0.2)
        x = self.dropout(x)
        
        x = self.conv4(x)
        x = F.leaky_relu(self.bn4(x), 0.2)
        
        # Flatten and pass through linear layer
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        
        return x

def weights_init(m):
    """
    Initialize network weights for better training stability.
    Args:
        m: Module to initialize weights for
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def compress_model(model, compression_ratio=0.5):
    """
    Compress model by reducing the number of parameters.
    Args:
        model: PyTorch model to compress
        compression_ratio: Ratio of parameters to keep (0.5 = 50% reduction)
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            # Reduce number of filters by compression_ratio
            if hasattr(module, 'out_channels'):
                new_out_channels = int(module.out_channels * compression_ratio)
                if new_out_channels > 0:
                    module.out_channels = new_out_channels
        elif isinstance(module, nn.Linear):
            # Reduce number of features by compression_ratio
            if hasattr(module, 'out_features'):
                new_out_features = int(module.out_features * compression_ratio)
                if new_out_features > 0:
                    module.out_features = new_out_features
    return model 