# Input-agnostic neural network model

In computer vision, a **2D input** usually refers to an image with dimensions:

<pre>
(B, C, H, W)
</pre>

- **B** – batch size  
- **C** – number of channels (e.g., 3 for RGB)  
- **H, W** – image height and width  

Many early CNN architectures (e.g., AlexNet, VGG, original ResNet) were trained on **fixed-size images** such as `224×224`.  
This requirement comes from the existence of fully connected (dense) layers at the end of the network, which expect a specific flattened input size.


## Why Fixed-size Inputs Are Limiting

1. Real-world images vary in size
2. Resizing can distort information
3. Segmentation: For example, in semantic segmentation, the model must produce an output aligned with the input, e.g., `(H, W)` → `(H, W)`.

## How Modern Architectures Handle Variable Sizes

Instead of FC layers at the end, modern CNNs use:

- Global Pooling Layers that collapse `(H, W)` into `(1,1)` regardless of input size.
  - e.g., `AdaptiveAvgPool2d((1,1))` in PyTorch

- Fully Convolutional Networks (FCN). With FCN, the network remains purely convolutional accepting any `(H, W)`.
  - e.g., Replace FC layers with **1×1 convolutions**.

- Upsampling / Decoders for Segmentation. 
  - Models like **U-Net** or **DeepLabv3** upsample feature maps back to `(H, W)` so predictions match the input size.

## Why We Need Variable-size Inputs

- Flexibility: Train once, deploy on different input sizes.  
- Accuracy: Preserve spatial detail instead of forcing all data into a single resolution.  
- Efficiency: Process data at its natural size (e.g., avoid resizing very large images unnecessarily).  

