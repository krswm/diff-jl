using SparseArrays
using Statistics

using Flux  # For `conv`.
using JSON
using SafeTensors

include("model.jl")
using .Model

function tshow(x)
    println(size(x))
    show(IOContext(stdout, :limit => true), "text/plain", x)
    println()
end

function norm_inner(x, g, t, x_mean, x_var)
    # x [...]
    # g (scalar)
    # t (scalar)
    # x_mean (scalar)
    # x_var (scalar)
    g .* (x .- x_mean) ./ √(x_var + oftype(x_var, 1.0f-5)) .+ t  # [...]
end

function groupnorm(x, g, t)
    # x [x, y, o, n]
    # g [o]
    # t [o]
    num_x, num_y, num_o, num_n = size(x)
    num_g = 32
    num_o ÷ num_g == 0
    # j: Index inside a group
    num_j = num_o ÷ num_g

    y = reshape(x, (num_x, num_y, num_j, num_g, num_n))  # [x, y, j, g, n]
    y_mean = mean(y, dims=(1, 2, 3))  # [x, y, j, g, n] (x=1, y=1, j=1 only)
    y_mean = [y_mean[1, 1, 1, (o - 1) ÷ num_j + 1, n] for o=1:num_o, n=1:num_n]  # [o, n]
    y_var = var(y, dims=(1, 2, 3))  # [x, y, j, g, n] (x=1, y=1, j=1 only)
    y_var = [y_var[1, 1, 1, (o - 1) ÷ num_j + 1, n] for o=1:num_o, n=1:num_n]  # [o, n]
    x = eachslice(x, dims=(3, 4))  # [o, n][x, y]
    x = [norm_inner(x[o, n], g[o], t[o], y_mean[o, n], y_var[o, n]) for o=1:num_o, n=1:num_n]  # [o, n][x, y]
    stack(x)  # [x, y, o, n]
end

silu(x) = x / (exp(-x) + 1)

function calc_drblock(x, drblock)
    y = x  # [x, y, o, n]
    x = groupnorm(x, drblock.g1, drblock.t1)  # [x, y, o, n]
    x = silu.(x)  # [x, y, o, n]
    x = conv(x, drblock.wc1, stride=1, pad=1, flipped=true) .+ drblock.bc1
    x = groupnorm(x, drblock.g2, drblock.t2)  # [x, y, o, n]
    x = silu.(x)  # [x, y, o, n]
    x = conv(x, drblock.wc2, stride=1, pad=1, flipped=true) .+ drblock.bc2
    y + x
end

function decode(x, dmodel)
    @assert eltype(x) == Float32
    @assert size(x) == (1, 4, 64, 64)  # [n, o, y, x]
    x = permutedims(x, (4, 3, 2, 1))  # [x, y, o, n]
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 4, 1)
    x ./= 0.18215f0
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 4, 1)
    x = conv(x, dmodel.dconv_pq.wc, stride=1, pad=0, flipped=true) .+ dmodel.dconv_pq.bc
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 4, 1)
    x = conv(x, dmodel.dconv_in.wc, stride=1, pad=1, flipped=true) .+ dmodel.dconv_in.bc
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 512, 1)
    x = calc_drblock(x, dmodel.drblock_mid1)
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 512, 1)
    x |> tshow
end

decref = load_safetensors("../../../Downloads/decref.safetensors")
dmodel = begin
    tensors = load_safetensors("../../../Downloads/sd/v1-5/model.safetensors")
    get_dmodel(tensors)
end
x = decref["l"]
decode(x, dmodel)
