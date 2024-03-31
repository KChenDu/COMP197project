# Solution 1

To write a encoder that adapts to the SMP decoder, we need a encoder that output the same format as the decoder input. The original MiT encoder output is an array of 4D tensors, and we need to adheare to that format.

The encoder output dimension shape is defined as follows:

| Index | Shape | Explain   | Description   |
|-------|-------|-----------|---------------|
| 0 | 16x3x224x224 | Batch_Size * 3_Color_Channels * height * width | Original Image Tensor |
|1| 16x0x112x112 | Batch_Size * 0_Channels * height/2 * width/2 | Dummy Tensor |
|2| 16x64x56x56 | Batch_Size * 64_Channels * height/4 * width/4 | 1st ViT Block Output |
|3| 16x128x28x28 | Batch_Size * 128_Channels * height/8 * width/8 | 2nd ViT Block Output |
|4| 16x320x14x14 | Batch_Size * 320_Channels * height/16 * width/16 | 3rd ViT Block Output |
|5| 16x512x7x7 | Batch_Size * 512_Channels * height/32 * width/32 | 4th ViT Block Output |

Compare to traditional ViT encoder, each block of MiT does not accepting a same tensor shape, but each block accepting them at different shapes. That means, the output of each block was not being Upsampled or Downsampled from a fixed size tensor like 16x128x14x14 to a different output shape. Instead, the transformation of the tensor is done by a patch embedding layer that comes after each ViT block, and each block takes the output of the patch embedding (which shape is already being transformed) layer as input.

Hence that their MiT encoder model structure looks like this in each layer:

| Layer | Input Shape | Output Shape | Description |
|-------|-------------|--------------|---------|
| 0 | 16x3x224x224 | 16x3x224x224 | Original Image Input |
| 1 | 16x3x224x224 | 16x64x56x56 | Patch Embedding 0 |
| 2 | 16x64x56x56 | 16x64x56x56 | 1st ViT Block Output |
| 3 | 16x64x56x56 | 16x128x28x28 | Patch Embedding 1 |
| 4 | 16x128x28x28 | 16x128x28x28 | 2nd ViT Block Output |
| 5 | 16x128x28x28 | 16x320x14x14 | Patch Embedding 2 |
| 6 | 16x320x14x14 | 16x320x14x14 | 3rd ViT Block Output |
| 7 | 16x320x14x14 | 16x512x7x7 | Patch Embedding 3 |
| 8 | 16x512x7x7 | 16x512x7x7 | 4th ViT Block Output |

But since we are using traditional ViT encoder, we need to adapt the output to the MiT decoder input. We can adapt the output by Upsampling or Downsampling the tensor to the desired dimension. For our models, we can have such a model structure for ViT encoder:

| Layer | Input Shape | Output Shape | Description |
|-------|-------------|--------------|---------|
| 0 | 16x3x224x224 | 16x3x224x224 | Original Image Input |
| 1 | 16x3x224x224 | 16x320x14x14 | Patch Embedding |
| 2 | 16x320x14x14 | 16x320x14x14 | 1st ViT Block |
| 2-1 | 16x320x14x14 | 16x64x56x56 | Upsampling 4x Output |
| 3 | 16x320x14x14 | 16x320x14x14 | 2nd ViT Block |
| 3-1 | 16x320x14x14 | 16x128x28x28 | Upsampling 2x Output |
| 4 | 16x320x14x14 | 16x320x14x14 | 3rd ViT Block |
| 4-1 | 16x320x14x14 | 16x320x14x14 | Identity Output |
| 5 | 16x320x14x14 | 16x320x14x14 | 4th ViT Block |
| 5-1 | 16x320x14x14 | 16x512x7x7 | Downsampling 2x Output |

However, we have not verified the performance of such a model structure. This might not be as effective as the original MiT encoder. But I believe the performance will still be acceptable as a segmentation model.