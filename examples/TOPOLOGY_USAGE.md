# Cluster Topology Configuration Guide

This guide explains how to pass cluster topology (which models to run and where) to `launch.py`.

## What is Cluster Topology?

Cluster topology defines:
- **device_name**: The SLURM device/machine to use (e.g., `l40-8`, `a40-8`)
  - **NEW**: Supports wildcards! Use `*-8` for all 8-GPU machines, `l40*-8` for l40 variants, etc.
- **script_spec**: Reference to a pre-existing model configuration (e.g., `llama8b`, `skyworks8b`)
  - OR -
- **inline_config**: Define the model configuration directly inline
- **target_instances**: How many instances to run on this device type

## Two Ways to Specify Topology

### Option 1: YAML Configuration File (Recommended)

Create a YAML file like `topology.yaml`:

```yaml
configurations:
  - device_name: "a40-8"
    script_spec: "llama8b"
    target_instances: 8
  
  - device_name: "l40-8"
    script_spec: "llama8b"
    target_instances: 8
  
  - device_name: "l40s-8"
    script_spec: "skyworks8b"
    target_instances: 4
```

Then run:
```bash
python launch.py --mode=cluster --topology_file=topology.yaml
```

Or with absolute path:
```bash
python launch.py --mode=cluster --topology_file=/gscratch/ark/graf/literegistry-core/examples/topology.yaml
```

### Option 2: Default (Hardcoded)

If you don't provide any topology configuration, it will use the default hardcoded values in `launch.py`:

```bash
python launch.py --mode=cluster
```

## Available Devices

Available device configurations (from `machines.yaml`):
- Single GPU: `l40-1`, `l40s-1`, `a40-1`
- 2 GPUs: `l40-2`, `l40s-2`, `a40-2`
- 4 GPUs: `l40-4`, `l40s-4`, `a40-4`
- 8 GPUs: `l40-8`, `l40s-8`, `a40-8`

### Wildcard Device Matching (NEW!)

Instead of listing multiple devices, use wildcards:

| Pattern | Matches | Description |
|---------|---------|-------------|
| `*-4` | `a40-4`, `l40-4`, `l40s-4` | All 4-GPU machines |
| `*-8` | `a40-8`, `l40-8`, `l40s-8` | All 8-GPU machines |
| `l40*-8` | `l40-8`, `l40s-8` | All L40 variants with 8 GPUs |
| `a40-*` | `a40-1`, `a40-2`, `a40-4`, `a40-8` | All A40 machines |
| `*-1` | `a40-1`, `l40-1`, `l40s-1` | All single-GPU machines |

**Example:**
```yaml
configurations:
  # OLD WAY - had to list each one:
  # - device_name: "a40-4"
  #   ...
  # - device_name: "l40-4"
  #   ...
  # - device_name: "l40s-4"
  #   ...
  
  # NEW WAY - just use wildcard:
  - device_name: "*-4"
    target_instances: 4
    inline_config:
      script_type: vllm
      args:
        model: "meta-llama/Llama-3.1-8B-Instruct"
        tensor_parallel_size: ${DEVICE_COUNT}
        gpu_memory_utilization: 0.95
        max-model-len: 4096
        dtype: bfloat16
      launcher: literegistry
```

The script will automatically expand `*-4` into separate configurations for `a40-4`, `l40-4`, and `l40s-4`!

## Available Script Specs

Available model configurations:
- `llama8b`: Llama-3.1-8B-Instruct
- `skyworks8b`: Skyworks model
- (Add your own in the `configs` dict in `main()`)

## Inline Model Configuration (NEW!)

Instead of referencing pre-existing config files, you can now define vLLM model configurations **directly in the topology file**. This is useful when you want to quickly test different models without creating separate YAML files.

### Inline Config Structure

Replace `script_spec` with `inline_config` containing:

```yaml
inline_config:
  script_type: vllm  # Type of launcher (vllm, etc.)
  args:
    model: "model-name"  # HuggingFace model name
    tensor_parallel_size: ${DEVICE_COUNT}  # Auto-filled based on device
    gpu_memory_utilization: 0.95
    max-model-len: 4096
    dtype: bfloat16
  launcher: literegistry  # Launcher to use
```

### Example: Inline YAML Configuration

```yaml
configurations:
  - device_name: "l40-8"
    target_instances: 8
    inline_config:
      script_type: vllm
      args:
        model: "meta-llama/Llama-3.1-8B-Instruct"
        tensor_parallel_size: ${DEVICE_COUNT}
        gpu_memory_utilization: 0.95
        max-model-len: 4096
        dtype: bfloat16
      launcher: literegistry
  
  - device_name: "a40-8"
    target_instances: 4
    inline_config:
      script_type: vllm
      args:
        model: "meta-llama/Llama-3.1-70B-Instruct"
        tensor_parallel_size: ${DEVICE_COUNT}
        gpu_memory_utilization: 0.90
        max-model-len: 8192
        dtype: bfloat16
      launcher: literegistry
```


### Mixing Inline and Referenced Configs

You can mix both approaches in the same topology file:

```yaml
configurations:
  # Use inline config for a custom model
  - device_name: "l40-8"
    target_instances: 8
    inline_config:
      script_type: vllm
      args:
        model: "mistralai/Mistral-7B-Instruct-v0.2"
        tensor_parallel_size: ${DEVICE_COUNT}
        gpu_memory_utilization: 0.95
        max-model-len: 4096
        dtype: bfloat16
      launcher: literegistry
  
  # Reference existing config file
  - device_name: "a40-8"
    target_instances: 8
    script_spec: "llama8b"
```

### Common vLLM Arguments

- **model**: HuggingFace model identifier (e.g., `"meta-llama/Llama-3.1-8B-Instruct"`)
- **tensor_parallel_size**: Number of GPUs for tensor parallelism (use `${DEVICE_COUNT}` to auto-fill)
- **gpu_memory_utilization**: Fraction of GPU memory to use (0.0 to 1.0)
- **max-model-len**: Maximum sequence length
- **dtype**: Data type (`bfloat16`, `float16`, `float32`)
- See [vLLM docs](https://docs.vllm.ai/) for more options

## Example Use Cases

### Run same model on different GPUs:
```yaml
configurations:
  - device_name: "l40-8"
    script_spec: "llama8b"
    target_instances: 8
  - device_name: "a40-8"
    script_spec: "llama8b"
    target_instances: 8
```

### Run different models on different GPUs:
```yaml
configurations:
  - device_name: "l40-8"
    script_spec: "llama8b"
    target_instances: 8
  - device_name: "a40-4"
    script_spec: "skyworks8b"
    target_instances: 4
```

### Mix single and multi-GPU setups:
```yaml
configurations:
  - device_name: "l40-1"
    script_spec: "llama8b"
    target_instances: 16
  - device_name: "l40-8"
    script_spec: "llama8b"
    target_instances: 2
```

## Example Files Provided

- `topology.yaml` - Basic topology with script_spec references
- `topology_inline.yaml` - Topology with inline vLLM configs
- `topology_wildcard.yaml` - Topology using wildcard device names
- `topology_simple.yaml` - Simplest possible example

## Other Launch Options

All original launch.py options still work:

```bash
# Terminate all running jobs
python launch.py --mode=terminate

# Run a single job interactively
python launch.py --mode=run --device=l40-8 --spec=llama8b

# Change account or user
python launch.py --mode=cluster --topology_file=topology.yaml --account=myaccount --user=myuser
```

