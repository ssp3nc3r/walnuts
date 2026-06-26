# WALNUTS Setup and Installation Guide

This guide documents how to install WALNUTS (a C++ HMC sampler) and use it with Stan models via BridgeStan.

This is a fork of [flatironinstitute/walnuts](https://github.com/flatironinstitute/walnuts) with additional CLI features on the `stan-cli-diagnostics` branch: JSON initialization, diagnostic output, unit mass matrix option, and additional sampler configuration arguments.

## Overview

WALNUTS is an alternative sampler to Stan's default NUTS. It requires:
1. A compiled Stan model as a shared library (.so file) via BridgeStan
2. Data in JSON format
3. (Optional) Initial values in JSON format

## Prerequisites

- C++ compiler (clang or gcc with C++17 support)
- CMake (3.14+)
- Git
- R with bridgestan package (for compiling Stan models to .so files)

## Step 1: Clone and Build WALNUTS

Clone the repository and build the `stan_cli` executable:

```bash
git clone -b stan-cli-diagnostics https://github.com/ssp3nc3r/walnuts.git
cd walnuts
mkdir build && cd build
cmake -DWALNUTS_BUILD_STAN=ON ..
make stan_cli
```

This creates the executable at `walnuts/build/examples/stan_cli`.

### Using from other project directories

The built binary can be called from any project directory using its path:

```bash
path/to/walnuts/build/examples/stan_cli \
  --samples 1000 \
  --warmup 1000 \
  --refresh 1 \
  --output draws.csv \
  /absolute/path/to/model_model.so \
  /absolute/path/to/dat.json
```

Use absolute paths for the model `.so` file and data files when running from a different directory.

## Step 2: Compile Your Stan Model to a Shared Library

Use the `bridgestan` R package to compile your model to a `.so` shared library:

```r
library(bridgestan)

# Compile the model (with optional threading support)
mod_path <- compile_model(
  "path/to/your/model.stan",
  make_args = list("STAN_THREADS=true")
)

# The .so file will be created at:
# path/to/your/model_model.so
```

**Note:** `cmdstanr::cmdstan_model(..., compile_standalone = TRUE)` does NOT create `.so` files - it creates executables. You must use `bridgestan::compile_model()` for WALNUTS.

Install bridgestan if needed:

```r
remotes::install_github("https://github.com/roualdes/bridgestan", subdir="R")
```

## Step 3: Prepare Your Data as JSON

Your Stan data needs to be in JSON format. In R:

```r
library(jsonlite)

# Assuming 'dat' is your Stan data list
json_data <- toJSON(dat, auto_unbox = TRUE, digits = NA)
write(json_data, "dat.json")
```

## Step 4: (Optional) Prepare Initial Values as JSON

If your model needs specific initial values (recommended for complex models):

```r
library(jsonlite)

# Your init function that returns a named list
init_from_priors <- function(dat, chain = 1) {
  # ... your initialization logic ...
  list(
    param1 = value1,
    param2 = matrix(...),
    # etc.
  )
}

# Generate and save inits
inits <- init_from_priors(dat, chain = 1)
json_inits <- toJSON(inits, auto_unbox = TRUE, digits = NA)
write(json_inits, "inits.json")
```

## Step 5: Run WALNUTS

Basic usage:

```bash
path/to/walnuts/build/examples/stan_cli \
  --samples 1000 \
  --warmup 1000 \
  --refresh 1 \
  --output draws.csv \
  path/to/model_model.so \
  path/to/dat.json
```

With custom initialization:

```bash
path/to/walnuts/build/examples/stan_cli \
  --samples 1000 \
  --warmup 1000 \
  --refresh 1 \
  --step-learning-rate 0.02 \
  --step-size-init 0.01 \
  --step-accept-rate-target 0.8 \
  --max-error 0.3 \
  --init-json path/to/inits.json \
  --output draws.csv \
  path/to/model_model.so \
  path/to/dat.json
```

With Stan-style unit mass matrix initialization (recommended if sampler gets stuck):

```bash
path/to/walnuts/build/examples/stan_cli \
  --samples 1000 \
  --warmup 1000 \
  --refresh 1 \
  --unit-mass \
  --init-json path/to/inits.json \
  --output draws.csv \
  path/to/model_model.so \
  path/to/dat.json
```

## CLI Options Reference

```
--seed              Random seed (default: random from clock)
--warmup            Number of warmup iterations (default: 128)
--samples           Number of samples to draw (default: 128)
--refresh           Iterations between progress reports (default: 100)
--max-depth         Maximum NUTS trajectory depth (default: 10)
--max-step-depth    Maximum step size adaptation depth (default: 8)
--min-micro-steps   Minimum micro steps per macro step (default: 1)
--max-error         Maximum error in joint densities (default: 0.5)
--init              Range for random init [-init, init] (default: 2.0)
--init-json         Path to JSON file with initial values
--output            Output CSV file for draws
--unit-mass         Use unit (identity) mass matrix initialization (Stan-style)
                    instead of gradient-based initialization (Nutpie-style)
--step-size-init    Initial step size (default: 1.0)
--step-accept-rate-target  Target acceptance rate (default: 0.8)
--step-learning-rate       Learning rate for step adaptation (default: 0.2)
--step-beta1        Adam beta1 (default: 0.3)
--step-beta2        Adam beta2 (default: 0.99)
--step-epsilon      Adam epsilon (default: 1e-4)
--target-depth      Target trajectory doublings (default: 3.5)
```

### Mass Matrix Initialization Options

WALNUTS supports two mass matrix initialization strategies:

1. **Nutpie-style (default)**: Uses `sqrt(|gradient|)` at the initial position to estimate parameter scales. This can be more efficient when gradients are well-behaved, but may cause problems if gradients vary wildly in scale.

2. **Stan-style (`--unit-mass`)**: Uses identity (all ones) mass matrix initialization. This is more conservative and matches Stan's default behavior. Recommended if the sampler gets stuck or the step size becomes extremely small during warmup.

Both options still adapt the mass matrix during warmup - the difference is only in the starting point for adaptation.

## Output Files

- `draws.csv` - Posterior draws (parameters as columns, draws as rows)
- `draws_diagnostics.csv` - Mass matrix diagonal (for large models)

## Troubleshooting

### "model: File does not exist"
- Use absolute paths for the model .so file and data files
- Positional arguments (model, data) must come AFTER all --options

### "Error in logp_grad" messages during warmup
- These are normal when the sampler explores bad regions of parameter space
- If they persist, check your model for numerical stability issues
- Consider using custom initialization to start from a good region

### "Initial log probability is not finite"
- Your initial values are rejected by the model
- Check constraints in your Stan model
- Generate new initial values that satisfy all constraints

### Sampler gets stuck or step size becomes very small
- The Nutpie-style gradient-based mass matrix initialization may not work well if gradients vary wildly in scale
- Try adding `--unit-mass` to use Stan-style identity mass matrix initialization
- This is more conservative but avoids initialization problems with badly-scaled gradients

## Notes

- WALNUTS uses Adam optimizer for step size adaptation (more robust than dual averaging)
- Automatic tuning for minimum micro steps per macro step
- Single-threaded sampler (run multiple instances for multiple chains)
- Mass matrix is diagonal only (no dense mass matrix support)
