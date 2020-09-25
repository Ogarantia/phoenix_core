# Upstride Engine


### Environment variables

There is a bunch of environment variables controlling the runtime Engine behavior.

**UPSTRIDE_VERBOSE** enables the Engine debugging messages with `UPSTRIDE_SAYS(..)` macro (when set to `1`). This is only effective if `UPSTRIDE_ALLOW_VERBOSE` compilation flag is set up. For a production release this flag is intended to be disabled in order to avoid debugging messages appearing in a readable form in the Engine binaries. It is therefore safe to print sensitive information.

**UPSTRIDE_MEMORY_OPTIMIZED** enables selecting a memory-efficient implementation to a speed-optimized alternative when possible (when set to `1`). Namely,
* It enables the default quaternion convolution implementation performing 16 small convolutions with quaternions components instead the factorized implementation doing 8 small convolutions but requiring more intermediate memory.

**UPSTRIDE_CONV_FP16_POLICY** specifies the convolution computing precision for 16-bit floating point inputs and outputs.
* When set to `full16`, the convolution is computed with 16-bit half precision (fast, inaccurate).
* When set to `backward32`, the forward pass is computed in 16-bit half precision, whereas the backward pass is computed in 32-bit single precision (intermediate).
* When set to anything else or unspecified, the convolution is computed with 32-bit single precision. This is default behavior of TensorFlow 2.3 (slow, accurate).


### Compilation switches

`UPSTRIDE_ALLOW_VERBOSE` (default: ON) enables verbose messages with `UPSTRIDE_SAYS(..)`. This is intended to be switched off in production releases.

`UPSTRIDE_ENABLE_FP16` (default: ON) enables half-precision floating point support on GPU. This requires an NVidia GPU having CUDA compute capability of at least 5.3 at runtime. A binary compiled with `UPSTRIDE_ENABLE_FP16` and run on a device having CUDA compute capability less than 5.3 fails.