# %% [markdown]
# GPT-2 inference with Julia
# Copyright (C) 2026  Kurosawa Mutsumi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# %%
println("hey!")

# %%
using JSON
using SafeTensors
using Statistics

# %%
include("model.jl")
using .Model
include("tokenizer.jl")
using .Tokenizer
include("transformer.jl")
using .Transformer

# %%
function get_prompt_embedding(ids::Vector{Int}, model::Model.Model)::Matrix{Float32}
    k_caches = [
        [Matrix{Float32}(undef, model.n_embd ÷ model.n_head, 0) for _ = 1:model.n_head]
        for _ = 1:model.n_layer
    ]
    v_caches = [
        [Matrix{Float32}(undef, model.n_embd ÷ model.n_head, 0) for _ = 1:model.n_head]
        for _ = 1:model.n_layer
    ]
    x = Vector{Float32}[]
    for (pos, id) ∈ enumerate(ids)
        push!(x, transformer!(id, pos, model, k_caches, v_caches))
    end
    x = hcat(x...)
    x
end

# %%
#=
if length(ARGS) ≠ 4
    println("Machine learning stuff with Julia")
    print("Usage: ")
    printstyled(
        "julia --project $PROGRAM_FILE <path to model repository> <path to pre-sampled random tensors> <your positive prompt> <your negative prompt>",
        bold = true,
    )
    println()
    println("You may have to enclose 'your prompt' with quotes.")
    exit()
end
=#
ARGS1 = "../../../Downloads/sd/v1-5"
ARGS2 = "../../../Downloads/rand42.safetensors"
ARGS3 = "cat with hat"
ARGS4 = ""

token_to_id, id_to_token = begin
    vocab = JSON.parsefile("$(ARGS1)/vocab.json")
    token_to_id = Dict(token => id for (token, id) ∈ vocab)
    id_to_token = Dict(id => token for (token, id) ∈ vocab)
    token_to_id, id_to_token
end

ranks = begin
    ranks = Dict{Tuple{String,String},Int}()
    rank = 0
    for line ∈ readlines("$(ARGS1)/merges.txt")
        # Skip a comment line.
        if startswith(line, "#")
            continue
        end

        token0, token1 = split(line, " ")
        ranks[(token0, token1)] = rank
        rank += 1
    end
    ranks
end

model = begin
    tensors = load_safetensors("$(ARGS1)/model.safetensors")
    config = JSON.parsefile("$(ARGS1)/config.json")
    get_model(tensors, config)
end

# ==== Tokenization ====

# Token IDs
positive_ids = tokenize(token_to_id, ranks, model, ARGS3)
negative_ids = tokenize(token_to_id, ranks, model, ARGS4)

# ==== Inference ====

positive_prompt_embedding = get_prompt_embedding(positive_ids, model)
negative_prompt_embedding = get_prompt_embedding(negative_ids, model)
c = cat(positive_prompt_embedding, negative_prompt_embedding, dims = 3)

# ====

function conv2d(wc::Array{Float32, 4}, bc::Vector{Float32}, latent::Array{Float32, 4})::Array{Float32, 4}
    # Kernel size 3x3, padding 1

    N, Cin, H, W = size(latent)

    Cout, Cin_, HH, WW = size(wc)
    @assert Cin == Cin_ && HH == WW && HH % 2 == 1

    kw = HH ÷ 2
    # kernelwidth = 0
    #
    # .....
    # .....
    # ..O..
    # .....
    # .....
    #
    # kernelwidth = 1
    #
    # .....
    # .OOO.
    # .OOO.
    # .OOO.
    # .....

    _X_s = eachslice(permutedims(latent, (2, 1, 3, 4)), dims=(3, 4))
    # Matrix of matrices(↓)
    #
    # X[i=1 n=1] X[i=1 n=2] ...
    # X[i=2 n=1] X[i=2 n=2]
    # ...                   ...
    #
    # for each (y, x)
    
    # model.enc_wc1
    #               Cout Cin η ξ

    _W_s = eachslice(wc, dims=(3, 4))
    # Matrix of matrices(↓)
    #
    # W[o=1 i=1] W[o=1 i=2] ...
    # W[o=2 i=1] W[o=2 i=2]
    # ...                   ...
    #
    # for each (η, ξ)

    _A_ = [_W_s[η, ξ] * _X_s[y, x] for η = 1:HH, ξ = 1:WW, y = 1:H, x = 1:W]
    # 4D tensor of matrices(↓)
    #
    # A[o=1 n=1] A[o=1 n=2] ...
    # A[o=2 n=1] A[o=2 n=2]
    # ...                   ...
    #
    # for each (η, ξ, y, x)

    [
        sum(
            1 ≤ y + Δy ≤ H && 1 ≤ x + Δx ≤ W
            ? _A_[Δy + kw + 1, Δx + kw + 1, y + Δy, x + Δx][o, n] : 0.0f0
            for Δy = -kw:kw, Δx = -kw:kw
        ) + bc[o]
        for n = 1:N, o = 1:Cout, y = 1:H, x = 1:W
    ]
end

function groupnorm(g::Vector{Float32}, t::Vector{Float32}, num_groups::Int, conv::Array{Float32, 4})::Array{Float32, 4}
    N, C, H, W = size(conv)
    @assert C % num_groups == 0
    size_of_group = C ÷ num_groups
    
    conv_mean = [
        mean(
            conv[n, group * size_of_group + igroup, y, x]
            for igroup = 1:size_of_group, y = 1:H, x = 1:W
        )
        for n = 1:N, group = 0:(num_groups - 1)
    ]

    conv_var = [
        var(
            (
                conv[n, group * size_of_group + igroup, y, x]
                for igroup = 1:size_of_group, y = 1:H, x = 1:W
            ), corrected = false
        )
        for n = 1:N, group = 0:(num_groups - 1)
    ]

    [
        (g[o] * (conv[n, o, y, x] - conv_mean[n, (o - 1) ÷ size_of_group + 1]) / √(conv_var[n, (o - 1) ÷ size_of_group + 1] + 1f-5) + t[o])
        for n = 1:N, o = 1:C, y = 1:H, x = 1:W
    ]
end

function layer_norm(x, g, t)
    g .* (x .- mean(x)) ./ √(var(x, corrected = false) + 1f-5) + t
end

# Pre-sampled random tensors
# The diffusion model requires a random noise,
# however, I want the whole tensor calculation deterministic
# so that I can compare the result with the reference implementation.
# I extracted the random tensors from reference implementation 
# (https://github.com/hkproj/pytorch-stable-diffusion, excellent explanation video!)
# and save it to a Safetensors file.
# I'll use genuine random number generator in Julia later.
rand42 = load_safetensors(ARGS2)

latent = rand42["l"]

t = 900

f = t .* 10000 .^ (0.0f0:(-1.0f0/160):(-159.0f0/160))
f = vcat(cos.(f), sin.(f))

f = model.time_w1 * f + model.time_b1
f = f ./ (exp.(-f) .+ 1)
f = model.time_w2 * f + model.time_b2

# Indices:
# - n (batch)   1 ≤ n ≤ 2
# - i (Cin)     1 ≤ i ≤ 4
# - o (Cout)    1 ≤ o ≤ 320
# - y (y axis)  1 ≤ y ≤ 64
# - x (x axis)  1 ≤ x ≤ 64
# - η (Δy+2)    1 ≤ η ≤ 3
# - ξ (Δx+2)    1 ≤ ξ ≤ 3

latent = cat(latent, latent, dims = 1)
# \/
# /\ n i y x

latent = conv2d(model.enc_wc1, model.enc_bc1, latent)


####

# conv
#      n o y x
# - n (batch)   1 ≤ n ≤ 2
# - o (Cout)    1 ≤ o ≤ 320
# - y (y axis)  1 ≤ y ≤ 64
# - x (x axis)  1 ≤ x ≤ 64


#=
r = reshape(conv, (64 * 64 * 10, 32 * 2))

rr = [
    model.enc_gg1 .* (r[:, gn] .- mean(r[:, gn])) ./ √(var(r[:, gn], corrected = false) + 1f-5) + model.enc_tg1
    for gn = 1:(32 * 2)
]

rr |> size |> println
rr[1] |> size |> println

rrr = reshape(rr, (64, 64, 320, 2))
=#

x = groupnorm(model.enc_gg1, model.enc_tg1, 32, latent)

x = x ./ (exp.(-x) .+ 1)

x = conv2d(model.enc_wc2, model.enc_bc2, x)

f = f ./ (exp.(-f) .+ 1)

f = model.enc_time_w1 * f + model.enc_time_b1

merged = [
    x[n, o, y, x_] + f[o]
    for n = 1:2, o = 1:320, y = 1:64, x_ = 1:64
]
merged = groupnorm(model.g_1_0_out_layers_0, model.t_1_0_out_layers_0, 32, merged)
merged = merged ./ (exp.(-merged) .+ 1)
merged = conv2d(model.wc_1_0_out_layers_3, model.bc_1_0_out_layers_3, merged)
latent += merged

x = latent
x = groupnorm(model.g_1_1_norm, model.t_1_1_norm, 32, x)
x = conv2d(model.wc_1_1_proj_in, model.bc_1_1_proj_in, x)

# n o y x -> x y o n -> xy o n -> n [xy o]
# (Pytorch is row-major but Julia is column-major)
x = permutedims(x, (4, 3, 2, 1))
x = reshape(x, (64 * 64, 320, 2))
x = eachslice(x, dims=3)
y = x

x = [
    begin
        yy = [
            layer_norm(collect(z), model.g_1_1_transformer_blocks_0_norm1, model.t_1_1_transformer_blocks_0_norm1)
            for z in eachslice(y, dims=1)
        ]
        hcat(yy...)
    end for y in x
]

x[1]

# %%

# %%

# %%
"hello!"

# %%
# n [xy o]



# Self attention.
# Difference from the attention for GPT-2 or CLiP:
# - No bias on input projection. Only weight matrix.
# - No causal mask (that means I can't use KV-cache)


# Numerically stable softmax
function softmax(x)
    x = exp.(x .- maximum(x))
    x / sum(x)
end


n_embd = 320
n_head = 8
size_head = n_embd ÷ n_head
x1 = [
    begin
        y = model.w_1_1_transformer_blocks_0_attn1 * y
        qq = y[1:n_embd, :]
        kk = y[(n_embd + 1):(2 * n_embd), :]
        vv = y[(2 * n_embd + 1):(3 * n_embd), :]
        q = (qq[((i - 1) * size_head + 1):(i * size_head), :] for i = 1:n_head)
        k = (kk[((i - 1) * size_head + 1):(i * size_head), :] for i = 1:n_head)
        v = (vv[((i - 1) * size_head + 1):(i * size_head), :] for i = 1:n_head)
        kq = transpose.(k) .* q ./ sqrt(Float32(size_head))
        kqq = [hcat([softmax(kq__) for kq__ in eachcol(kq_)]...) for kq_ in kq]
        v .* kqq
    end
    for y ∈ x
]

x1[1][1]

# %%
size_head

# %%
x2 = [
      model.w_1_1_transformer_blocks_0_attn1_to_out_0 * [
        x1[n][(o - 1) ÷ size_head + 1][(o - 1) % size_head + 1, xy]
        for o = 1:320
    ]  + model.b_1_1_transformer_blocks_0_attn1_to_out_0

    for n = 1:2, xy = 1:4096
]

# %%

# %%
# x1  # [n][head][ihead, xy]
a1 = [x1[n][head][ihead, xy] for n=1:2, xy=1:4096, ihead=1:40, head=1:8]  # [n, xy, ihead, head]
a2 = reshape(a1, (2, 4096, 320))  # [n, xy, o]
a3 = (a2[n, xy, :] for n=1:2, xy=1:4096)  # [n, xy][o]
a4 = Ref(model.w_1_1_transformer_blocks_0_attn1_to_out_0) .* a3 .+ Ref(model.b_1_1_transformer_blocks_0_attn1_to_out_0)  # [n, xy][o]
x2 = (a4[n, xy][o] for n=1:2, xy=1:4096, o=1:320)  # [n, xy, o]

# %%
_y = (y[n][xy, o] for n=1:2, xy=1:4096, o=1:320)  # [n, xy, o]
x4 = _y .+ x2  # [n, xy, o]

# %%
a1 = (x4[n, xy, :] for n=1:2, xy=1:4096)  # [n, xy][o]
a2 = layer_norm.(a1, Ref(model.g_1_1_transformer_blocks_0_norm2), Ref(model.t_1_1_transformer_blocks_0_norm2))  # [n, xy][o]
x5 = (a[n, xy][o] for n=1:2, xy=1:4096, o=1:320)  # [n, xy, o]
x5 |> collect
