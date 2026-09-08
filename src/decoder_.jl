using DelimitedFiles
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

function calc_drcblock(x, drcblock)
    y = x  # [x, y, o, n]
    x = groupnorm(x, drcblock.g1, drcblock.t1)  # [x, y, o, n]
    x = silu.(x)  # [x, y, o, n]
    x = conv(x, drcblock.wc1, stride=1, pad=1, flipped=true) .+ drcblock.bc1
    x = groupnorm(x, drcblock.g2, drcblock.t2)  # [x, y, o, n]
    x = silu.(x)  # [x, y, o, n]
    x = conv(x, drcblock.wc2, stride=1, pad=1, flipped=true) .+ drcblock.bc2
    x + (conv(y, drcblock.wc3, stride=1, pad=0, flipped=true) .+ drcblock.bc3)
end

function self_attention(x, w1, b1, w2, b2, num_h)
    # x [xy, o]
    # w1 [o, 3o]
    # b1 [1, 3o]
    # w2 [o, o]
    # b2 [1, o]
    # It's a little unfortunate that I have to use xᵀ Wᵀ + bᵀ for an affine transformation
    # because of Julia using column-major (leftmost index changes the fastest)
    # (If I understand it correctly, though. Maybe I'm missing something and totally wrong)
    # rather than W x + b that is more "intuitive" for me
    # as a person that learn it with the latter form in linear algebra class.
    num_xy, num_o = size(x)
    num_j = num_o ÷ num_h  # Size of a head

    # 1 ≤ 3o ≤ 3 * num_o
    x = x * w1 .+ b1  # [xy, 3o]
    # 1 ≤ qkv ≤ 3
    x = reshape(x, (num_xy, num_j, num_h, 3))  # [xy, j, h, qkv]
    q = eachslice(x[:, :, :, 1], dims=3)  #[h][xy, j]
    k = eachslice(x[:, :, :, 2], dims=3)  #[h][xy', j]
    v = eachslice(x[:, :, :, 3], dims=3)  #[h][xy, j]
    qk = @. q * transpose.(k) / √Float32(num_j)  # [h][xy, xy']
    qk = softmax.(qk, dims=2)  # [h][xy', xy]
    x = qk .* v  # [h][xy, j]
    x = stack(x)  # [xy, j, h]
    x = reshape(x, (num_xy, num_o))  # [xy, o]
    x * w2 .+ b2  # [xy, o]
end

function calc_dablock(x, dablock)
    # x [x, y, o, n]
    num_x, num_y, num_o, num_n = size(x)
    num_xy = num_x * num_y

    y = x  # [x, y, o, n]

    x = groupnorm(x, dablock.g, dablock.t)  # [x, y, o, n]

    x = reshape(x, (num_xy, num_o, num_n))  # [xy, o, n]
    x = eachslice(x, dims=3)  # [n][xy, o]


    # g [o]
    # t [o]

    x = self_attention.(x, Ref(dablock.w1), Ref(dablock.b1), Ref(dablock.w2), Ref(dablock.b2), 1)  # [n][xy, o]
    
    x = stack(x)  # [xy, o, n]
    x = reshape(x, (num_x, num_y, num_o, num_n))  # [x, y, o, n]
    y += x
    y
end

function upsample(x)
    # x [half_x, half_y, o, n]
    num_half_x, num_half_y, num_o, num_n = size(x)
    @views [x[x_ ÷ 2 + 1, y ÷ 2 + 1, o, n] for x_=0:(2 * num_half_x - 1), y=0:(2 * num_half_y - 1), o=1:num_o, n=1:num_n] # [x, y, o, n]
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

    @time x = conv(x, dmodel.dconv_pq.wc, stride=1, pad=0, flipped=true) .+ dmodel.dconv_pq.bc
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 4, 1)
    @time x = conv(x, dmodel.dconv_in.wc, stride=1, pad=1, flipped=true) .+ dmodel.dconv_in.bc
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 512, 1)

    @time x = calc_drblock(x, dmodel.drblock_mid1)
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 512, 1)
    @time x = calc_dablock(x, dmodel.dablock)
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 512, 1)
    @time x = calc_drblock(x, dmodel.drblock_mid2)
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 512, 1)

    @time x = calc_drblock(x, dmodel.drblock_30)
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 512, 1)
    @time x = calc_drblock(x, dmodel.drblock_31)
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 512, 1)
    @time x = calc_drblock(x, dmodel.drblock_32)
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 512, 1)
    @time x = upsample(x)
    @assert eltype(x) == Float32
    @assert size(x) == (128, 128, 512, 1)
    @time x = conv(x, dmodel.dconv_3.wc, stride=1, pad=1, flipped=true) .+ dmodel.dconv_3.bc
    @assert eltype(x) == Float32
    @assert size(x) == (128, 128, 512, 1)

    @time x = calc_drblock(x, dmodel.drblock_20)
    @time x = calc_drblock(x, dmodel.drblock_21)
    @time x = calc_drblock(x, dmodel.drblock_22)
    @time x = upsample(x)
    @time x = conv(x, dmodel.dconv_2.wc, stride=1, pad=1, flipped=true) .+ dmodel.dconv_2.bc

    @time x = calc_drcblock(x, dmodel.drcblock_10)
    @time x = calc_drblock(x, dmodel.drblock_11)
    @time x = calc_drblock(x, dmodel.drblock_12)
    @time x = upsample(x)
    @time x = conv(x, dmodel.dconv_1.wc, stride=1, pad=1, flipped=true) .+ dmodel.dconv_1.bc

    @time x = calc_drcblock(x, dmodel.drcblock_00)
    @time x = calc_drblock(x, dmodel.drblock_01)
    @time x = calc_drblock(x, dmodel.drblock_02)

    x = groupnorm(x, dmodel.dgn.g, dmodel.dgn.t)
    x = silu.(x)
    @time x = conv(x, dmodel.dconv_out.wc, stride=1, pad=1, flipped=true) .+ dmodel.dconv_out.bc

    # Expected result!
    # Thank you Flux.jl for providing me a fast 2D convolution implementation.
    
    x
end

#=
decref = load_safetensors("../../../Downloads/decref.safetensors")
dmodel = begin
    tensors = load_safetensors("../../../Downloads/sd/v1-5/model.safetensors")
    get_dmodel(tensors)
end
x = decref["l"]
x = decode(x, dmodel)

x = clamp.(floor.((x .+ 1) * 128), UInt8)  # [x, y, rgb, n]

function generate_ppm_image(x, filename)
    println(size(x))
    num_x, num_y, num_rgb, num_n = size(x)
    @assert num_rgb == 3
    @assert num_n == 1
    x = permutedims(x, (4, 3, 1, 2))  # [n, rgb, x, y]
    x = vec(x)  # [rgbyx]
    open(filename, "w") do file
        println(file, "P3", " ", num_x, " ", num_y, " ", 255)
        writedlm(file, x)
    end
end

generate_ppm_image(x, ARGS[1])
=#

function calc_frblock(x, f, frblock)
    y = x  # [x, y, o, n]
    x = groupnorm(x, frblock.g1, frblock.t1)  # [x, y, o, n]
    x = silu.(x)  # [x, y, o, n]
    x = conv(x, frblock.wc1, stride=1, pad=1, flipped=true) .+ frblock.bc1
    f = silu.(f)  # [f₃]
    f = frblock.w * f .+ frblock.b  # [o]
    f = insertdims(f, dims=(1, 2, 4))  # [x, y, o, n]
    x .+= f  # [x, y, o, n]
    x = groupnorm(x, frblock.g2, frblock.t2)  # [x, y, o, n]
    x = silu.(x)  # [x, y, o, n]
    x = conv(x, frblock.wc2, stride=1, pad=1, flipped=true) .+ frblock.bc2
    x + y
end

function calc_frcblock(x, f, frcblock)
    y = x  # [x, y, o, n]
    x = groupnorm(x, frcblock.g1, frcblock.t1)  # [x, y, o, n]
    x = silu.(x)  # [x, y, o, n]
    x = conv(x, frcblock.wc1, stride=1, pad=1, flipped=true) .+ frcblock.bc1
    f = silu.(f)  # [f₃]
    f = frcblock.w * f .+ frcblock.b  # [o]
    f = insertdims(f, dims=(1, 2, 4))  # [x, y, o, n]
    x .+= f  # [x, y, o, n]
    x = groupnorm(x, frcblock.g2, frcblock.t2)  # [x, y, o, n]
    x = silu.(x)  # [x, y, o, n]
    x = conv(x, frcblock.wc2, stride=1, pad=1, flipped=true) .+ frcblock.bc2
    x + (conv(y, frcblock.wc3, stride=1, pad=0, flipped=true) .+ frcblock.bc3)
end

function denoise(x, c, t, prev_t, fmodel)
    println("==== t = $t ====")

    f = t .* 10000 .^ (0.0f0:(-1.0f0/160):(-159.0f0/160))  # [f₁]
    f = vcat(cos.(f), sin.(f))  # [f₁]

    f = fmodel.time_w1 * f + fmodel.time_b1  # [f₂]
    f = silu.(f)  # [f₂]
    f = fmodel.time_w2 * f + fmodel.time_b2  # [f₃]

    x = cat(x, x, dims = 4)  # [x, y, o, n]

    print("0.0 ")
    @time x = conv(x, fmodel.fconv_i0.wc, stride=1, pad=1, flipped=true) .+ fmodel.fconv_i0.bc
    s0 = x
    print("1.0 ")
    @time x = calc_frblock(x, f, fmodel.frblock_i1)
    x |> tshow
end

function diffuse(c, fmodel)
    rand42 = load_safetensors("../../../Downloads/rand42.safetensors")
    x = permutedims(rand42["l"], (4, 3, 2, 1))  # [x, y, o, n]

    x = denoise(x, c, 900, 800, fmodel)
end

decref = load_safetensors("../../../Downloads/decref.safetensors")
fmodel = begin
    tensors = load_safetensors("../../../Downloads/sd/v1-5/model.safetensors")
    get_fmodel(tensors)
end
c = permutedims(decref["c"], (3, 2, 1))  # [ctx, idx, n]
diffuse(c, fmodel)
