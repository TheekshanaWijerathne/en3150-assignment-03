# Custom architectures

*Both models take a 3×64×64 image and return one raw score per class.*

| Model | Standard conv | Depthwise conv | Pointwise conv | Fully connected | Activation | Trainable parameters | MACs per image |
|---|---:|---:|---:|---:|---|---:|---:|
| Model A | 3 | 0 | 0 | 2 | ReLU | 1,143,466 | 42,337,536 |
| Model B | 1 | 4 | 4 | 1 | ReLU6 | 71,818 | 12,989,568 |


## Model A - main layers

| Layer | Type | Kernel | Filters / units | Stride | Output | Parameters |
|---|---|---|---|---|---|---:|
| 0.0 | Conv2d | 3×3 | 32 | 1 | 32×64×64 | 864 |
| 2.0 | Conv2d | 3×3 | 64 | 1 | 64×32×32 | 18,432 |
| 4.0 | Conv2d | 3×3 | 128 | 1 | 128×16×16 | 73,728 |
| 7 | Linear |  | 128 |  | 128 | 1,048,704 |
| 10 | Linear |  | 10 |  | 10 | 1,290 |


## Model B - main layers

| Layer | Type | Kernel | Filters / units | Stride | Output | Parameters |
|---|---|---|---|---|---|---:|
| 0.0 | Conv2d | 3×3 | 16 | 1 | 16×64×64 | 432 |
| 1.0 | Conv2d depthwise | 3×3 | 16 | 1 | 16×64×64 | 144 |
| 1.3 | Conv2d pointwise | 1×1 | 32 | 1 | 32×64×64 | 512 |
| 3.0 | Conv2d depthwise | 3×3 | 32 | 1 | 32×32×32 | 288 |
| 3.3 | Conv2d pointwise | 1×1 | 64 | 1 | 64×32×32 | 2,048 |
| 5.0 | Conv2d depthwise | 3×3 | 64 | 1 | 64×16×16 | 576 |
| 5.3 | Conv2d pointwise | 1×1 | 160 | 1 | 160×16×16 | 10,240 |
| 7.0 | Conv2d depthwise | 3×3 | 160 | 1 | 160×8×8 | 1,440 |
| 7.3 | Conv2d pointwise | 1×1 | 320 | 1 | 320×8×8 | 51,200 |
| 11 | Linear |  | 10 |  | 10 | 3,210 |
