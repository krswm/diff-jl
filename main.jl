# Encoder

# Is there `nn.Conv2d` equivalent in Julia?
# `conv2`? `UndefVarError` happens for me??
# No longer existing in base Julia?
# DSP.jl?
using DSP

# `GroupNorm` in Julia?
# Do I have to write it myself?
# Looking at the formula, GroupNorm looks similar to LayerNorm.
# What is different?

# `SiLU`?
σ(x) = 1 / (exp(-x) + 1)  # Fermi distribution flipped
silu(x) = x * σ(x)
# using Plots
# plot(silu) |> display
# readline()

# Decoder
