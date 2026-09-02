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
        global rank  # Temporary workaronud. I'll remove it when I wrap the code with `main` again.

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

function conv2d(wc, bc, latent)
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

    # latent [n, i, y, x]
    _X_s = [[latent[n, i, y, x] for i=1:Cin, n=1:N] for y=1:H, x=1:W]  # [y, x][i, n]
    # Matrix of matrices(↓)
    #
    # X[i=1 n=1] X[i=1 n=2] ...
    # X[i=2 n=1] X[i=2 n=2]
    # ...                   ...
    #
    # for each (y, x)
    
    # model.enc_wc1
    #               Cout Cin η ξ

    # wc [o, i, η, ξ]
    _W_s = [wc[:, :, η, ξ] for η=1:HH, ξ=1:WW]
    # Matrix of matrices(↓)
    #
    # W[o=1 i=1] W[o=1 i=2] ...
    # W[o=2 i=1] W[o=2 i=2]
    # ...                   ...
    #
    # for each (η, ξ)

    _A_ = [_W_s[η, ξ] * _X_s[y, x] for η = 1:HH, ξ = 1:WW, y = 1:H, x = 1:W]  # [η, ξ, y, x][o, n]
    # 4D tensor of matrices(↓)
    #
    # A[o=1 n=1] A[o=1 n=2] ...
    # A[o=2 n=1] A[o=2 n=2]
    # ...                   ...
    #
    # for each (η, ξ, y, x)

    # _A_ [η, ξ, y, x][o, n]
    # sum_A_ [y, x, n][o]
    # bc [o]

    # [n, o, y, x]

    (
        sum(
            1 ≤ y + Δy ≤ H && 1 ≤ x + Δx ≤ W
            ? _A_[Δy + kw + 1, Δx + kw + 1, y + Δy, x + Δx][o, n] : 0.0f0
            for Δy = -kw:kw, Δx = -kw:kw
        ) + bc[o]
        for n = 1:N, o = 1:Cout, y = 1:H, x = 1:W
    )  # [n, o, y, x]
end

function groupnorm(g, t, num_groups, conv)
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

#### RESIDUAL BLOCK ####

####

# conv
#      n o y x
# - n (batch)   1 ≤ n ≤ 2
# - o (Cout)    1 ≤ o ≤ 320
# - y (y axis)  1 ≤ y ≤ 64
# - x (x axis)  1 ≤ x ≤ 64


function calc_rblock(latent, f, rblock)
    num_n, num_o, num_y, num_x = size(latent)

    x = groupnorm(rblock.g1, rblock.t1, 32, latent)  # [n, o, y, x]
    x = x ./ (exp.(-x) .+ 1)  # [n, o, y, x]
    x = conv2d(rblock.wc1, rblock.bc1, x) |> collect  # [n, o, y, x]
    f = f ./ (exp.(-f) .+ 1)  # [o]
    f = rblock.w * f + rblock.b  # [o]
    _x_ = (x[n, :, y, x_] for n=1:num_n, y=1:num_y, x_=1:num_x)  # [n, y, x][o]
    merged = Ref(f) .+ _x_  # [n, y, x][o]
    merged = [merged[n, y, x][o] for n=1:num_n, o=1:num_o, y=1:num_y, x=1:num_x]  # [n, o, y, x]
    merged = groupnorm(rblock.g2, rblock.t2, 32, merged)  # [n, o, y, x]
    merged = merged ./ (exp.(-merged) .+ 1)  # [n, o, y, x]
    merged = conv2d(rblock.wc2, rblock.bc2, merged)  # [n, o, y, x]
    latent .+ merged  # [n, o, y, x]
end

#### ATTENTION BLOCK ####
 
# Numerically stable softmax
function softmax(x)
    x = exp.(x .- maximum(x))
    x / sum(x)
end

function calc_ablock(x, c, ablock)
    num_n, num_o, num_y, num_x = size(latent)
    num_xy = num_x * num_y

    x = latent  # [n, o, y, x]
    x = groupnorm(ablock.g1, ablock.t1, 32, x)  # [n, o, y, x]
    x = conv2d(ablock.wc1, ablock.bc1, x) |> collect  # [n, o, y, x]

    x = [x[n, o, y_, x_] for n=1:num_n, x_=1:num_x, y_=1:num_y, o=1:num_o]  # [n, x, y, o]
    # (Pytorch is row-major but Julia is column-major)
    x = reshape(x, (num_n, num_xy, num_o))  # [n, xy, o]
    x = [x[n, :, :] for n=1:num_n]  # [xy, o][n]
    y = x  # [xy, o][n]

    x = [
        begin
            yy = [
                layer_norm(collect(z), ablock.g2, ablock.t2)
                for z in eachslice(y, dims=1)
            ]
            hcat(yy...)
        end for y in x
    ]

    # Self attention.
    # Difference from the attention for GPT-2 or CLiP:
    # - No bias on input projection. Only weight matrix.
    # - No causal mask (that means I can't use KV-cache)

    n_embd = num_o
    n_head = 8
    size_head = n_embd ÷ n_head
    x1 = [
        begin
            y = ablock.w21 * y
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

    # x1  # [n][head][ihead, xy]
    a1 = [x1[n][head][ihead, xy] for n=1:num_n, xy=1:num_xy, ihead=1:size_head, head=1:n_head]  # [n, xy, ihead, head]
    a2 = reshape(a1, (num_n, num_xy, num_o))  # [n, xy, o]
    a3 = (a2[n, xy, :] for n=1:num_n, xy=1:num_xy)  # [n, xy][o]
    a4 = Ref(ablock.w22) .* a3 .+ Ref(ablock.b22)  # [n, xy][o]
    x2 = [a4[n, xy][o] for n=1:num_n, xy=1:num_xy, o=1:num_o]  # [n, xy, o]

    # %%
    _y = (y[n][xy, o] for n=1:num_n, xy=1:num_xy, o=1:num_o)  # [n, xy, o]
    x4 = _y .+ x2  # [n, xy, o]

    # %%
    a1 = (x4[n, xy, :] for n=1:num_n, xy=1:num_xy)  # [n, xy][o]
    a2 = layer_norm.(a1, Ref(ablock.g3), Ref(ablock.t3))  # [n, xy][o]
    x5 = [a2[n, xy][o] for n=1:num_n, xy=1:num_xy, o=1:num_o]  # [n, xy, o]

    c  # [embd, ctx, n]

    num_embd, num_ctx, _ = size(c)

    _c = [c[embd, ctx, n] for n=1:num_n, ctx=1:num_ctx, embd=1:num_embd]  # [n, ctx, embd]
    ;

    # %%
    a = (x5[n, xy, :] for n=1:num_n, xy=1:num_xy)  # [n, xy][o]
    a = Ref(ablock.w31q) .* a  # [n, xy][o]
    q_attn2 = [a[n, xy][o] for n=1:num_n, xy=1:num_xy, o=1:num_o]  # [n, xy, o]
    ;


    # %%
    a = (_c[n, ctx, :] for n=1:num_n, ctx=1:num_ctx)  # [n, ctx][embd]
    a = Ref(ablock.w31k) .* a  # [n, ctx][o]
    k_attn2 = [a[n, ctx][o] for n=1:num_n, ctx=1:num_ctx, o=1:num_o]  # [n, ctx, o]
    ;

    # %%
    a = (_c[n, ctx, :] for n=1:num_n, ctx=1:num_ctx)  # [n, ctx][embd]
    a = Ref(ablock.w31v) .* a  # [n, ctx][o]
    v_attn2 = [a[n, ctx][o] for n=1:num_n, ctx=1:num_ctx, o=1:num_o]  # [n, ctx, o]
    ;

    # %%
    q_attn2_r = reshape(q_attn2, (num_n, num_xy,  size_head, n_head))  # [n, xy, ihead, head]
    k_attn2_r = reshape(k_attn2, (num_n, num_ctx, size_head, n_head))  # [n, ctx, ihead, head]
    v_attn2_r = reshape(v_attn2, (num_n, num_ctx, size_head, n_head))  # [n, ctx, ihead, head]
    ;

    # %%
    x7 = [k_attn2_r[n, :, :, head] * q_attn2_r[n, xy, :, head] / sqrt(Float32(size_head)) for n=1:num_n, xy=1:num_xy, head=1:n_head]  # [n, xy, head][ctx]
    ;

    # %%
    x8 = softmax.(x7)  # [n, xy, head][ctx]
    ;

    # %%
    x9 = [v_attn2_r[n, :, :, head]' * x8[n, xy, head] for n=1:num_n, xy=1:num_xy, head=1:n_head]  # [n, xy, head][ihead]
    ;

    # %%
    a = [x9[n, xy, head][ihead] for n = 1:num_n, xy=1:num_xy, ihead=1:size_head, head=1:n_head]
    x10 = reshape(a, (num_n, num_xy, num_o))  # [n, xy, o]
    x11 = [x10[n, xy, :] for n=1:num_n, xy=1:num_xy]
    ;

    # %%
    a = Ref(ablock.w32) .* x11 .+ Ref(ablock.b32)  # [n, xy][o]
    x12 = [a[n, xy][o] for n=1:num_n, xy=1:num_xy, o=1:num_o]  # [n, xy, o]
    ;

    # %%
    x13 = x4 + x12  # [n, xy, o]
    ;

    # %%
    x13 = Float32.(x13)
    ;

    # %%
    ugelu(u, v) = u .* (tanh.((v .^ 3 * 0.044715f0 + v) * sqrt(2.0f0 / pi)) .+ 1.0f0) .* v .* 0.5f0
    # `u` and `v` are vectors.

    # %%
    a = (x13[n, xy, :] for n=1:num_n, xy=1:num_xy)  # [n, xy][o]
    a = layer_norm.(a, Ref(ablock.g4), Ref(ablock.t4))  # [n, xy][o]
    a = Ref(ablock.w41) .* a .+ Ref(ablock.b41) #[n, xy][o8] (1 <= o8 <= 4 * 320 * 2)
    a = reshape.(a, Ref((4 * num_o, 2)))  # [n, xy][o4, chunk]  (1 <= o4 <= 4 * 320, 1 <= chunk <= 2)
    b, c = ((a[n, xy][:, chunk] for n=1:num_n, xy=1:num_xy) for chunk=1:num_n)  # [n, xy][o4], [n, xy][o4]
    d = ugelu.(b, c)  # [n, xy][o4]
    a = Ref(ablock.w42) .* d .+ Ref(ablock.b42)  # [n, xy][o]
    d = (a[n, xy][o] for n=1:num_n, xy=1:num_xy, o=1:num_o)  # [n, xy, o]
    d = x13 .+ d  # [n, xy, o]
    # `.` in `.+` is necessary when I add an `Array` and a `Generator` of same shape.
    d = [d[n, xy, o] for n=1:num_n, o=1:num_o, xy=1:num_xy]  # [n, o, xy]
    d = reshape(d, (num_n, num_o, num_x, num_y))  # [n, o, x, y]
    a = [d[n, o, x, y] for n=1:num_n, o=1:num_o, y=1:num_y, x=1:num_x]  # [n, o, y, x]
    a = conv2d(ablock.wc4, ablock.bc4, a)  #[n, o, y, x]
    latent .+ a
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

println("~~~~ A ~~~~")

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

print("0.0 ")
latent = conv2d(model.enc_wc1, model.enc_bc1, latent) |> collect
print("1.0 ")
latent = calc_rblock(latent, f, model.rblocks["model.diffusion_model.input_blocks.1.0"])
print("1.1 ")
latent = calc_ablock(latent, c, model.ablocks["model.diffusion_model.input_blocks.1.1"])
print("2.0 ")
latent = calc_rblock(latent, f, model.rblocks["model.diffusion_model.input_blocks.2.0"])
print("2.1 ")
latent = calc_ablock(latent, c, model.ablocks["model.diffusion_model.input_blocks.2.1"])


show(IOContext(stdout, :limit => true), "text/plain", latent)
# Correct result!

# %% [markdown]
# I finished implementing the residual block and the attention block. Yay!
