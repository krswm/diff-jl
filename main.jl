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

# Attention

# Taken from my GPT-2 transformer implementation
# Numerically stable softmax
function softmax(x::Vector{Float32})::Vector{Float32}
    x = exp.(x .- maximum(x))
    x / sum(x, dims = 2)
end

# Taken from my GPT-2 transformer implementation
# This time I can't use KV-cache. (Maybe I can but not now.)
function multi_head_attention!(
    x::Matrix{Float32},
    n_head::Int,
    n_embd::Int,
    use_causal_mask::Bool,
)::Vector{Float32}
    # x = layer.w11 * x + layer.b11  # What is `nn.Linear`?
    chunks = Iterators.partition.(
        Iterators.partition(x, model.n_embd),
        model.n_embd ÷ model.n_head,
    )
    size_embd = model.n_embd ÷ model.n_head
    q = x[(0*size_embd):(1*size_embd), :]
    k = x[(1*size_embd):(2*size_embd), :]
    v = x[(2*size_embd):(3*size_embd), :]
    # Scaled dot-product attention
    a =
        v .* softmax.(
            transpose.(k) .* q ./ √Float32(size_embd) +
            triu(fill(-Inf, (size_embd, size_embd)), 1),
        )
    x = vcat(a...)
    # x = layer.w12 * x + layer.b12
    x
end
