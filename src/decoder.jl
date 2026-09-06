using SparseArrays
using Statistics

using JSON
using SafeTensors

include("model.jl")
using .Model

function tshow(x)
    println(size(x))
    show(IOContext(stdout, :limit => true), "text/plain", x)
    println()
end

function conv2d_3x3(I, F)
    # Kernel: 3x3
    num_x, num_y, num_i = size(I)  # [x, y, i]
    @assert num_x == num_y
    a = num_x
    num_ξ, num_η, num_i_ = size(F)  # [ξ, η, i]
    @assert num_ξ == num_η == 3
    
    FF_sp = sparse(
        vcat(  # rows
            (
                vcat(
                    ((2:a  ) .+ (c-1)*a for c=2:a  )...,
                    ((2:a  ) .+ (c-1)*a for c=1:a  )...,
                    ((2:a  ) .+ (c-1)*a for c=1:a-1)...,

                    ((1:a  ) .+ (c-1)*a for c=2:a  )...,
                    ((1:a  ) .+ (c-1)*a for c=1:a  )...,
                    ((1:a  ) .+ (c-1)*a for c=1:a-1)...,

                    ((1:a-1) .+ (c-1)*a for c=2:a  )...,
                    ((1:a-1) .+ (c-1)*a for c=1:a  )...,
                    ((1:a-1) .+ (c-1)*a for c=1:a-1)...,
                ) for i=1:num_i
            )...
        ),
        vcat(  # columns
            (
                vcat(
                    ((1:a-1) .+ (r-1)*a .+ (i-1)*a*a for r=1:a-1)...,
                    ((1:a-1) .+ (r-1)*a .+ (i-1)*a*a for r=1:a  )...,
                    ((1:a-1) .+ (r-1)*a .+ (i-1)*a*a for r=2:a  )...,

                    ((1:a  ) .+ (r-1)*a .+ (i-1)*a*a for r=1:a-1)...,
                    ((1:a  ) .+ (r-1)*a .+ (i-1)*a*a for r=1:a  )...,
                    ((1:a  ) .+ (r-1)*a .+ (i-1)*a*a for r=2:a  )...,

                    ((2:a  ) .+ (r-1)*a .+ (i-1)*a*a for r=1:a-1)...,
                    ((2:a  ) .+ (r-1)*a .+ (i-1)*a*a for r=1:a  )...,
                    ((2:a  ) .+ (r-1)*a .+ (i-1)*a*a for r=2:a  )...,
                ) for i=1:num_i
            )...
        ),
        vcat(
            (
                vcat(
                    fill(F[1, 1, i], (a-1)*(a-1)),  #  "10"
                    fill(F[1, 2, i], a*(a-1)    ),  #  "40"
                    fill(F[1, 3, i], (a-1)*(a-1)),  #  "70"

                    fill(F[2, 1, i], a*(a-1)    ),  #  "20"
                    fill(F[2, 2, i], a*a        ),  #  "50"
                    fill(F[2, 3, i], a*(a-1)    ),  #  "80"

                    fill(F[3, 1, i], (a-1)*(a-1)),  #  "30"
                    fill(F[3, 2, i], a*(a-1)    ),  #  "60"
                    fill(F[3, 3, i], (a-1)*(a-1)),  #  "90"
                ) for i=1:num_i
            )...
        ),
        a*a,
        a*a*num_i,
    ) |> dropzeros!

    II = vec(I)

    result = reshape(FF_sp * II, (a, a))  # [x, y]
    result
end

function conv2d_1x1(I, F)
    # kernel: 1x1
    num_x, num_y, num_i = size(I)  # [x, y, i]
    @assert num_x == num_y
    a = num_x
    num_ξ, num_η, num_i_ = size(F)  # [ξ, η, i]
    @assert num_ξ == num_η == 1
    
    FF_sp = sparse(
        vcat(  # rows
            (
                vcat(
                    ((1:a  ) .+ (c-1)*a for c=1:a  )...,
                ) for i=1:num_i
            )...
        ),
        vcat(  # columns
            (
                vcat(
                    ((1:a  ) .+ (r-1)*a .+ (i-1)*a*a for r=1:a  )...,
                ) for i=1:num_i
            )...
        ),
        vcat(
            (
                vcat(
                    fill(F[1, 1, i], a*a        ),  #  "50"
                ) for i=1:num_i
            )...
        ),
        a*a,
        a*a*num_i,
    ) |> dropzeros!

    II = vec(I)

    result = reshape(FF_sp * II, (a, a))  # [x, y]
    result
end

function conv2d(wc, bc, x)
    println("----")
    @time x = permutedims(x, (4, 3, 2, 1))
    @time wc = permutedims(wc, (4, 3, 2, 1))
    
    num_x, num_y, num_i, num_n = size(x)
    num_xy = num_x * num_y
    num_ξ, num_η, num_i_, num_o = size(wc)
    @assert num_i == num_i_
    @assert num_η == num_ξ
    @assert num_η % 2 == 1

    if num_ξ == 3
        wc = permutedims(wc, (2, 1, 3, 4))
        
        O = [conv2d_3x3(x[:, :, :, n], wc[:, :, :, o]) for o=1:num_o, n=1:num_n]  # [o, n][x, y]
        O = [O[o, n][x_, y] + bc[o] for x_=1:num_x, y=1:num_y, o=1:num_o, n=1:num_n]  # [x, y, o, n]

        O
    elseif num_ξ == 1
        wc = permutedims(wc, (2, 1, 3, 4))
        
        O = [conv2d_1x1(x[:, :, :, n], wc[:, :, :, o]) for o=1:num_o, n=1:num_n]  # [o, n][x, y]
        O = [O[o, n][x_, y] + bc[o] for x_=1:num_x, y=1:num_y, o=1:num_o, n=1:num_n]  # [x, y, o, n]

        O
    end
end

# TODO: Exactly the same as `groupnorm` in main.jl. Unify with it.
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

function normnorm(x, g, t, x_mean, x_var)
    # x [xy]
    # g (scalar)
    # t (scalar)
    # x_mean (scalar)
    # x_var (scalar)
    g .* (x .- x_mean) ./ √(x_var + oftype(x_var, 1.0f-5)) .+ t  # [xy]
end

function groupnorm_(x, g, t, num_g)
    # x [xy, o]
    # g [o]
    # t [o]
    # `num_g`: number of groups
    num_xy, num_o = size(x)
    @assert num_o % num_g == 0
    # j: Index inside a group
    num_j = num_o ÷ num_g

    y = reshape(x, (num_xy, num_j, num_g))  # [xy, j, g]
    y = eachslice(y, dims=3)  # [g][xy, j]
    y_mean = mean.(y)  # [g]
    y_mean = [y_mean[(o - 1) ÷ num_j + 1] for o ∈ 1:num_o]  # [o]
    y_var = var.(y)  # [g]
    y_var = [y_var[(o - 1) ÷ num_j + 1] for o ∈ 1:num_o]  # [o]
    x = eachslice(x, dims=2)  # [o][xy]
    x = normnorm.(x, g, t, y_mean, y_var)  # [o][xy]
    stack(x)  # [xy, o]
end

function layer_norm(x, g, t)
    g .* (x .- mean(x)) ./ √(var(x, corrected = false) + 1f-5) + t
end

function calc_drblock(latent, drblock)
    num_n, num_o, num_y, num_x = size(latent)  # latent [n, o, y, x]

    x = groupnorm(drblock.g1, drblock.t1, 32, latent)  # [n, o, y, x]
    x = x ./ (exp.(-x) .+ 1)  # [n, o, y, x]
    x = conv2d(drblock.wc1, drblock.bc1, x) |> collect  # [n, fo, y, x]
    _, num_fo, _, _ = size(x)
    x = groupnorm(drblock.g2, drblock.t2, 32, x)  # [n, fo, y, x]
    x = x ./ (exp.(-x) .+ 1)  # [n, fo, y, x]
    x = conv2d(drblock.wc2, drblock.bc2, x) |> collect  # [n, fo, y, x]
    if num_o == num_fo
        l = latent  # [n, fo, y, x]
    else
        l = conv2d(drblock.wc3, drblock.bc3, latent) |> collect  # [n, fo, y, x]
    end
    l .+ x  # [n, fo, y, x]
end

# Numerically stable softmax
function softmax(x; dims=2)
    x = exp.(x .- maximum(x, dims=dims))
    x ./ sum(x, dims=dims)
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
    # x [n, o, y, x]
    @time x = permutedims(x, (4, 3, 2, 1))  # [x, y, o, n]

    num_x, num_y, num_o, num_n = size(x)
    num_xy = num_x * num_y

    @time x = reshape(x, (num_xy, num_o, num_n))  # [xy, o, n]
    @time x = eachslice(x, dims=3)  # [n][xy, o]

    y = x

    # g [o]
    # t [o]
    @time x = groupnorm_.(x, Ref(dablock.g), Ref(dablock.t), 32)  # [n][xy, o]

    @time x = self_attention.(x, Ref(dablock.w1), Ref(dablock.b1), Ref(dablock.w2), Ref(dablock.b2), 1)  # [n][xy, o]

    @time y += x  # [n][xy, o]
    
    @time y = stack(y)  # [xy, o, n]
    @time y = reshape(y, (num_x, num_y, num_o, num_n))
    @time y = permutedims(y, (4, 3, 2, 1))
end

function upsample(x)
    # x [n, o, half_y, half_x]
    num_n, num_o, num_half_y, num_half_x = size(x)
    @views [x[n, o, y ÷ 2 + 1, x_ ÷ 2 + 1] for n=1:num_n, o=1:num_o, y=0:(2 * num_half_y - 1), x_=0:(2 * num_half_x - 1)] # [n, o, y, x]
end

decref = load_safetensors("../../../Downloads/decref.safetensors")

tensors = load_safetensors("../../../Downloads/sd/v1-5/model.safetensors")
config = JSON.parsefile("../../../Downloads/sd/v1-5/config.json")
model = get_model(tensors, config)

x = decref["l"]
@assert size(x) == (1, 4, 64, 64)
x ./= 0.18215

@assert size(x) == (1, 4, 64, 64)
@time x = conv2d(model.dconvs["first_stage_model.post_quant_conv"]..., x) |> collect
x |> tshow
exit()

@assert size(x) == (1, 4, 64, 64)
@time x = conv2d(model.dconvs["first_stage_model.decoder.conv_in"]..., x) |> collect
x |> tshow
exit()

@assert size(x) == (1, 512, 64, 64)

@time x = calc_drblock(x, model.drblocks["first_stage_model.decoder.mid.block_1"])
@assert size(x) == (1, 512, 64, 64)
@time x = calc_dablock(x, model.dablocks["first_stage_model.decoder.mid.attn_1"])
@assert size(x) == (1, 512, 64, 64)
x = calc_drblock(x, model.drblocks["first_stage_model.decoder.mid.block_2"])
@assert size(x) == (1, 512, 64, 64)

x = calc_drblock(x, model.drblocks["first_stage_model.decoder.up.3.block.0"])
@assert size(x) == (1, 512, 64, 64)
x = calc_drblock(x, model.drblocks["first_stage_model.decoder.up.3.block.1"])
@assert size(x) == (1, 512, 64, 64)
x = calc_drblock(x, model.drblocks["first_stage_model.decoder.up.3.block.2"])
@assert size(x) == (1, 512, 64, 64)

x = upsample(x)
@assert size(x) == (1, 512, 128, 128)
tshow(x)
