*Model B optimizer comparison.*

| Optimizer | Learning rate | Best epoch | Best validation accuracy (%) | Test accuracy (%) | Epoch to 90% of best | Mean epoch time (s) |
|---|---:|---:|---:|---:|---:|---:|
| SGD | 0.01 | 27 | 89.83 | 90.25 | 5 | 11.4 |
| SGD + momentum | 0.01 | 29 | 96.25 | 96.32 | 4 | 11.4 |
| Adam | 0.001 | 29 | 96.25 | 96.17 | 3 | 11.3 |
