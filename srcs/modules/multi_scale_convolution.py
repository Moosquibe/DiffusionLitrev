import torch

class MultiScaleConvolution(torch.nn.Module):
  """
  Makes use of long-range and multi-scale dependencies
  while generating a full resolution feature map using
  the following steps:
    1. Perform mean pooling to downsample the image to
       multiple scales.
    2. Perform convolution and a LeakyReLU activation
       at each scale (slightly different from the
       paper where there is a single activation in
       the very end).
    3. Upsample all scales to full resolution and
       sum the resulting images.

  Note that the convolution is applied parallel at
  each scale which could be done asynchronously.
  """
  def __init__(
      self,
      num_scales,
      in_channels,
      out_channels,
      kernel_size,
  ):
    self.convs = []
    self.acts = []

    self.num_scales = num_scales
    for _ in range(self.num_scales):
      self.convs.append(torch.nn.Conv2d(
          in_channels = in_channels,
          out_channels = out_channels,
          kernel_size = kernel_size,
          bias = True,
          padding = kernel_size - 1 # "full padding"
      ))
      self.activations.append([torch.nn.LeakyReLU(negative_slope=0.05)])

  def downsample(self, x, scale):
    """
    Downsamples an image by a factor of 2**scale
    collapsing pixel blocks by computing means.

    x : (B, C, H, W)
    """
    if scale == 0:
      return x

    B, C, H, W = x.shape()

    scale = 2 ** scale

    return (x
         .reshape(B, C, H // scale, scale, W // scale, scale)
         .mean(axis=5)
         .mean(axis=3)
    )

  def forward(self, x):
    """
    x: (B, C, H, W)
    """
    B, C, H, W = x.shape

    # TODO: 
    #   1. Input checking.
    #   2. Arithmetic for even size kernel.

    x_accum = 0
    overshoot = (self.kernel_size - 1) // 2
    for scale in range(self.num_scales-1, -1, -1):
      y = x.copy()
      y = self.downsample(y, scale)

      # Conv - Activation - Cropping back to pre-Conv size
      y = self.convs[scale](y)
      y = self.activations[scale](y)
      y = y[:, :, overshoot:-overshoot, overshoot:-overshoot]

      x_accum += y

      if scale > 0:
        # Upsample x_accum by a factor of 2 so that sizes match
        # in next iteration.
        H_scaled = H / 2 ** scale
        W_scaled = W / 2 ** scale
        
        x_accum = x_accum.reshape((B, C, H, 1, W, 1))
        x_accum = torch.cat([x_accum, x_accum], dim = 5)
        x_accum = torch.cat([x_accum, x_accum], dim = 3)
        x_accum = x_accum.reshape((B, C, H_scaled * 2, W_scaled * 2))

    return x_accum / self.num_scales