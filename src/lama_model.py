import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Core Component: Fast Fourier Convolution (FFC) Layer ---
# This is the key building block of the LaMa model. It processes local and global information simultaneously.

class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        
        # Calculate the number of channels for local and global processing
        self.in_channels_g = int(in_channels * ratio_gin)
        self.in_channels_l = in_channels - self.in_channels_g
        self.out_channels_g = int(out_channels * ratio_gout)
        self.out_channels_l = out_channels - self.out_channels_g

        # --- FIX: Ensure global channels are even ---
        # This is a requirement for the real/imaginary channel split.
        if self.in_channels_g % 2 != 0:
            self.in_channels_g -=1
            self.in_channels_l +=1
        if self.out_channels_g % 2 != 0:
            self.out_channels_g -= 1
            self.out_channels_l += 1
        # --- END FIX ---

        self.conv_l2l = nn.Conv2d(self.in_channels_l, self.out_channels_l, kernel_size,
                                  stride, padding, dilation, groups, bias)
        self.conv_g2l = nn.Conv2d(self.in_channels_g, self.out_channels_l, kernel_size,
                                  stride, padding, dilation, groups, bias)
        self.conv_l2g = nn.Conv2d(self.in_channels_l, self.out_channels_g, kernel_size,
                                  stride, padding, dilation, groups, bias)
                                  
        # --- FIX: The global conv layer now processes real and imaginary parts as channels ---
        # It needs twice the input and output channels.
        self.conv_g2g = nn.Conv2d(self.in_channels_g * 2, self.out_channels_g * 2, kernel_size,
                                  stride, padding, dilation, groups, bias)
        # --- END FIX ---


    def forward(self, x):
        x_l, x_g = x[:, :self.in_channels_l], x[:, self.in_channels_l:]

        out_l = self.conv_l2l(x_l) + self.conv_g2l(x_g)

        x_g_fft = torch.fft.rfft2(x_g, norm='ortho')
        x_g_real_imag = torch.view_as_real(x_g_fft)

        # --- FIX: Reshape the 5D tensor to 4D for the conv layer ---
        # Shape changes from [B, C, H, W, 2] to [B, C*2, H, W]
        b, c, h, w, _ = x_g_real_imag.shape
        x_g_reshaped = x_g_real_imag.permute(0, 1, 4, 2, 3).reshape(b, c * 2, h, w)
        # --- END FIX ---

        out_g_fft_real_imag = self.conv_g2g(x_g_reshaped)

        # --- FIX: Reshape the 4D output back to 5D ---
        # Shape changes from [B, C*2, H, W] back to [B, C, H, W, 2]
        b, _, h, w = out_g_fft_real_imag.shape
        out_g_reshaped = out_g_fft_real_imag.view(b, -1, 2, h, w).permute(0, 1, 3, 4, 2)
        # --- END FIX ---
        
        out_g_fft = torch.view_as_complex(out_g_reshaped.contiguous())
        
        out_g_ifft = torch.fft.irfft2(out_g_fft, s=x_g.size()[2:], norm='ortho')
        
        out_g = out_g_ifft + self.conv_l2g(x_l)

        return torch.cat((out_l, out_g), dim=1)


# --- FFC Residual Block ---
# This block is used repeatedly in the bottleneck of the LaMa generator.

class FFCResidualBlock(nn.Module):
    def __init__(self, channels, ratio_gin, ratio_gout):
        super().__init__()
        self.conv1 = FFC(channels, channels, kernel_size=3, padding=1, ratio_gin=ratio_gin, ratio_gout=ratio_gout)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = FFC(channels, channels, kernel_size=3, padding=1, ratio_gin=ratio_gin, ratio_gout=ratio_gout)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


# --- The Main LaMa Generator Model, Adapted for LST Inpainting ---

class LaMaGenerator(nn.Module):
    def __init__(self, input_channels, output_channels, num_ffc_blocks=9):
        super().__init__()
        
        self.initial_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, kernel_size=7, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.downsample1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        ffc_blocks = []
        for _ in range(num_ffc_blocks):
            ffc_blocks.append(FFCResidualBlock(256, ratio_gin=0.5, ratio_gout=0.5))
        self.ffc_blocks = nn.Sequential(*ffc_blocks)

        # Decoder: bilinear upsampling + conv (instead of ConvTranspose2d) to avoid checkerboard artifacts
        self.upsample1 = nn.Sequential(
            # 1. Bilinear Upsampling (Scaling) - No learnable parameters, purely geometric
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # 2. Standard Convolution - Learns the features on the upscaled image
            # Note: 'padding_mode="reflect"' is critical for LaMa to avoid edge artifacts
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect'), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, kernel_size=7, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.ffc_blocks(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.final_conv(x)
        return x

# --- Example of how to use the adapted model ---
if __name__ == '__main__':
    INPUT_CHANNELS = 21 
    OUTPUT_CHANNELS = 1

    model = LaMaGenerator(input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS)
    
    print("LaMa Generator Model for LST Inpainting Initialized")
    
    batch_size = 4
    height, width = 256, 256
    dummy_input = torch.randn(batch_size, INPUT_CHANNELS, height, width)
    
    print(f"\nTesting with a dummy input of shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    
    print(f"Output tensor shape: {output.shape}")
    
    assert output.shape == (batch_size, OUTPUT_CHANNELS, height, width)
    print("\nModel forward pass successful. Input and output dimensions are correct.") 

