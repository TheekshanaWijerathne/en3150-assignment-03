## Measuring what each model costs

None of these numbers depend on trained weights: an architecture costs the same to store and to run
whatever it has learned. So every model is built fresh, exactly as for training, and measured here,
all in one session on one machine.

- **Parameters**: trainable and total. Frozen pretrained weights still have to be stored.
- **Size on disk**: the saved weights (`state_dict`), as a device would store them, in KB
  (1 KB = 1,024 bytes). Saving them, rather than counting 4 bytes per parameter, includes buffers
  such as batch-normalisation statistics.
- **MACs** for one image, as counted above.
- **Latency**: milliseconds per image on **one CPU thread** at batch size 1, as a small edge device
  without a GPU would run it. 10 untimed passes first, then the mean of 100 timed passes; the median
  is recorded too.
- **Peak memory**: the weights plus the largest set of intermediate results alive at the same time
  during one forward pass, in MB (1 MB = 1,048,576 bytes). It is worked out from the model's graph
  (`torch.fx`): each intermediate result is freed after the last operation that uses it, and results
  that share memory (views, in-place operations) are counted once. It is therefore the same on any
  machine.

```python
import copy
import itertools
import statistics
import time

from torch import fx

LATENCY_THREADS = 1
LATENCY_WARMUP = 10
LATENCY_REPEATS = 100


def latency_samples_ms(model, input_shape=INPUT_SHAPE, warmup=LATENCY_WARMUP, repeats=LATENCY_REPEATS):
    """Milliseconds of each timed forward pass of one image, on one CPU thread.

    The untimed passes first pay one-off costs, such as allocating memory, that
    would otherwise inflate the result. The thread setting is restored afterwards.
    """
    probe = copy.deepcopy(model).cpu().eval()
    batch = torch.randn(1, *input_shape, generator=torch.Generator().manual_seed(0))
    threads = torch.get_num_threads()
    torch.set_num_threads(LATENCY_THREADS)
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                probe(batch)
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                probe(batch)
                samples.append((time.perf_counter() - started) * 1000)
    finally:
        torch.set_num_threads(threads)
    return samples


def tensors_in(value):
    """Every tensor inside a value, however nested in tuples, lists or dicts."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors_in(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors_in(item)


def weight_bytes(model):
    """Bytes of the parameters and buffers, each memory block counted once."""
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                for t in itertools.chain(model.parameters(), model.buffers())}
    return sum(storages.values())


def activation_timeline(model, input_shape=INPUT_SHAPE):
    """Bytes of intermediate results alive while each step of the model's graph runs.

    Every value is kept until the pass ends, so no memory address is reused: that
    is what lets results sharing memory (views, in-place operations) be recognised
    by their address. A result counts from the step that creates it to the last
    step that uses it.
    """
    graph_module = fx.symbolic_trace(model)
    nodes = list(graph_module.graph.nodes)
    position = {node: index for index, node in enumerate(nodes)}
    values = {}

    class Recorder(fx.Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            values[node] = value
            return value

    with torch.no_grad():
        Recorder(graph_module, garbage_collect_values=False).run(torch.zeros(1, *input_shape))

    weights = {t.untyped_storage().data_ptr()
               for t in itertools.chain(graph_module.parameters(), graph_module.buffers())}
    storages = {}  # memory address -> [bytes, first step, last step]
    end = len(nodes) - 1
    for node in nodes:
        if node.op == "get_attr":  # weights and constants, counted with the weights
            continue
        last = end if node.op == "output" else max(
            (position[user] for user in node.users), default=position[node])
        for tensor in tensors_in(values.get(node)):
            storage = tensor.untyped_storage()
            address = storage.data_ptr()
            if address in weights:
                continue
            if address in storages:  # a view or an in-place result: the same memory
                storages[address][2] = max(storages[address][2], last)
            else:
                storages[address] = [storage.nbytes(), position[node], last]
    return [sum(size for size, first, final in storages.values() if first <= step <= final)
            for step in range(len(nodes))]


def peak_memory_mb(model, input_shape=INPUT_SHAPE):
    """Weights plus the peak of live intermediate results, in MB."""
    probe = copy.deepcopy(model).cpu().eval()
    return (weight_bytes(probe) + max(activation_timeline(probe, input_shape))) / 2**20


def profile_model(name):
    """Every cost of the model ``name``, built fresh as for training."""
    model = build_model(name)
    samples = latency_samples_ms(model)
    return {
        "model": name,
        "trainable_params": count_parameters(model),
        "total_params": count_parameters(model, trainable_only=False),
        "model_size_kb": model_size_kb(model),
        "macs": count_macs(model),
        "latency_ms": statistics.fmean(samples),
        "latency_median_ms": statistics.median(samples),
        "peak_memory_mb": peak_memory_mb(model),
        "device": describe_device("cpu"),
        "latency_threads": LATENCY_THREADS,
        "latency_repeats": LATENCY_REPEATS,
    }
```
