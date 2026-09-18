# `artifacts/exports/` — deployment formats

**Git-ignored** (except this file). Optional — not required by the assignment.

If Member 4 exports an ONNX or TFLite model, it lands here. The size comparison against
`resources.json -> model_size_kb` makes a good Section 6 aside: int8 post-training quantisation
typically cuts model size about 4x, and Model B uses ReLU6 partly because its bounded output range
keeps that quantisation well-scaled.

Treat this as a bonus. Sections 1-6 are complete without it.
