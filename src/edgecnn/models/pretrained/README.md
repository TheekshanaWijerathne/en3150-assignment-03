# `models/pretrained/` — fine-tuned lightweight SOTA backbones

**Owner: Member 4.** Assignment §5 — **20 marks**.

## Inputs

`num_classes` and `input_shape` from the `DataBundle` — the **same splits** as the custom models,
which the assignment requires. Backbone settings from `configs/stages/pretrained.yaml`.

## Outputs

Two registry entries, `mobilenet_v2` and `squeezenet1_1`, registered into the **same registry** as
Model A and Model B. Member 3's trainer and Member 1's benchmark therefore need no special case for
them — which is exactly what makes the §6 comparison fair.

Each must also expose `param_groups()` so the trainer can apply a small learning rate to pretrained
weights and a full-sized one to the fresh head, without knowing anything about the architecture.

## Backbones

Both are from the list the assignment itself names. They reach parameter efficiency by *different*
mechanisms, and contrasting the mechanisms — not just the accuracy numbers — is what §6 rewards:

- **MobileNetV2** — depthwise separable convolutions with inverted residuals.
- **SqueezeNet 1.1** — fire modules: a 1x1 "squeeze" that cuts channel count, then a mixed 1x1/3x3
  "expand".

## Two traps

**Replacing the classifier.** They differ structurally:

- MobileNetV2 — `classifier[1]` is `Linear(1280, 1000)`
- SqueezeNet — `classifier[1]` is `Conv2d(512, 1000, 1)`, **not** a Linear layer, followed by
  global average pooling

Forgetting to replace MobileNetV2's head gives a model that trains without error and reports
nonsense, because the loss is computed over 1000 logits of which only 10 can ever be right.
Substituting a `Linear` into SqueezeNet fails on shape — that one at least fails loudly.

**Discriminative learning rates.** A head-sized learning rate applied to the backbone destroys the
pretrained features in the first few hundred steps. The symptom is the classic fine-tuning failure:
validation accuracy peaks at epoch 1 and then falls.

## The 64x64 decision

Fine-tuning happens at **64x64**, not at the backbones' native 224. This keeps the memory and
compute comparison honest — §6 is about cost at the resolution the sensor actually produces.

The handicap is real and is a **finding, not a flaw**: MobileNetV2 has a total stride of 32, so a
64x64 input reaches the classifier as a 2x2 spatial map where the architecture expects 7x7. A fair
report says so, rather than claiming a clean win for the custom model. Set
`inputs.input_resolution: 128` to run the ablation.

## For the report

Record **both** `trainable_params` and `total_params` — they differ whenever layers are frozen, and
a reader cannot interpret "2.2M parameters" without knowing how many actually moved. Also state
which freezing strategy was used, and whether ImageNet or EuroSAT normalisation statistics were
applied.
