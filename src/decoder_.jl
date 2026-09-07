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

function conv2d_1x1_inner(x_vec, wc, bc, a, num_i)
    # wc [x, y, i]
    wc_doubleblock = sparse(
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
                    fill(wc[1, 1, i], a*a        ),  #  "50"
                ) for i=1:num_i
            )...
        ),
        a*a,
        a*a*num_i,
    ) |> dropzeros!  # [α, αᵢ]

    # x_vec [αᵢ]
    # bc (scalar)
    result = wc_doubleblock * x_vec .+ bc  # [α]
    reshape(result, (a, a))  # [x, y]
end

function conv2d_1x1(x, wc, bc)
    @assert ndims(x) == 3    
    num_x, num_y, num_i = size(x)
    @assert num_x == num_y  # My implementation supports only square image currently.
    a = num_x

    @assert ndims(wc) == 4
    @assert eltype(wc) == eltype(x)
    num_ξ, num_η, num_i_, num_o = size(wc)
    @assert num_ξ == num_η == 1
    @assert num_i_ == num_i
    wc = eachslice(wc, dims=4)  # [o][ξ, η, i]

    @assert ndims(bc) == 1
    @assert eltype(bc) == eltype(x)
    num_o_, = size(bc)
    @assert num_o_ == num_o

    vec_x = vec(x)  # [αᵢ]

    result = conv2d_1x1_inner.(Ref(vec_x), wc, bc, a, num_i)  # [o][x, y]
    stack(result)
end

function conv2d_3x3_inner(x_vec, wc, bc, a, num_i)
    # wc [x, y, i]
    print("wc_doubleblock")
    @time wc_doubleblock = sparse(
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
                    fill(wc[1, 1, i], (a-1)*(a-1)),  #  "10"
                    fill(wc[1, 2, i], a*(a-1)    ),  #  "40"
                    fill(wc[1, 3, i], (a-1)*(a-1)),  #  "70"

                    fill(wc[2, 1, i], a*(a-1)    ),  #  "20"
                    fill(wc[2, 2, i], a*a        ),  #  "50"
                    fill(wc[2, 3, i], a*(a-1)    ),  #  "80"

                    fill(wc[3, 1, i], (a-1)*(a-1)),  #  "30"
                    fill(wc[3, 2, i], a*(a-1)    ),  #  "60"
                    fill(wc[3, 3, i], (a-1)*(a-1)),  #  "90"
                ) for i=1:num_i
            )...
        ),
        a*a,
        a*a*num_i,
    ) |> dropzeros!  # [α, αᵢ]

    # x_vec [αᵢ]
    # bc (scalar)
    print("result")
    @time result = wc_doubleblock * x_vec .+ bc  # [α]
    print("reshape")
    @time reshape(result, (a, a))  # [x, y]
end

function conv2d_3x3(x, wc, bc)
    @assert ndims(x) == 3    
    num_x, num_y, num_i = size(x)
    @assert num_x == num_y  # My implementation supports only square image currently.
    a = num_x

    @assert ndims(wc) == 4
    @assert eltype(wc) == eltype(x)
    num_ξ, num_η, num_i_, num_o = size(wc)
    @assert num_ξ == num_η == 3 "$(size(wc))"
    @assert num_i_ == num_i
    wc = eachslice(wc, dims=4)  # [o][ξ, η, i]

    @assert ndims(bc) == 1
    @assert eltype(bc) == eltype(x)
    num_o_, = size(bc)
    @assert num_o_ == num_o

    vec_x = vec(x)  # [αᵢ]

    result = conv2d_3x3_inner.(Ref(vec_x), wc, bc, a, num_i)  # [o][x, y]
    stack(result)
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
    # x [x, y, o]
    # g [o]
    # t [o]
    num_x, num_y, num_o = size(x)
    num_g = 32
    num_o ÷ num_g == 0
    # j: Index inside a group
    num_j = num_o ÷ num_g

    y = reshape(x, (num_x, num_y, num_j, num_g))  # [x, y, j, g]
    y_mean = mean(y, dims=(1, 2, 3))  # [g]
    y_mean = [y_mean[(o - 1) ÷ num_j + 1] for o ∈ 1:num_o]  # [o]
    y_var = var(y, dims=(1, 2, 3))  # [g]
    y_var = [y_var[(o - 1) ÷ num_j + 1] for o ∈ 1:num_o]  # [o]
    x = eachslice(x, dims=3)  # [o][x, y]
    x = norm_inner.(x, g, t, y_mean, y_var)  # [o][x, y]
    stack(x)  # [x, y, o]
end
    
#=
function groupnorm

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
=#

silu(x) = x / (exp(-x) + 1)

function calc_drblock(x, drblock)
    y = x  # [x, y, o]
    x = groupnorm(x, drblock.g1, drblock.t1)  # [x, y, o]
    x = silu.(x)  # [x, y, o]
    x = conv2d_3x3(x, drblock.wc1, drblock.bc1)  # [x, y, o]  # <- VERY slow :(
    x |> tshow
    y
end

function decode(x, dmodel)
    @assert eltype(x) == Float32
    @assert size(x) == (1, 4, 64, 64)
    x = reshape(x, (4, 64, 64))  # [o₀, y, x]
    x = permutedims(x, (3, 2, 1))  # [x, y, o₀]
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 4)
    x ./= 0.18215f0
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 4)
    @time x = conv2d_1x1(x, dmodel.dconv_pq.wc, dmodel.dconv_pq.bc)
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 4)
    @time x = conv2d_3x3(x, dmodel.dconv_in.wc, dmodel.dconv_in.bc)
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 512)
    x = calc_drblock(x, dmodel.drblock_mid1)
    @assert eltype(x) == Float32
end

decref = load_safetensors("../../../Downloads/decref.safetensors")
dmodel = begin
    tensors = load_safetensors("../../../Downloads/sd/v1-5/model.safetensors")
    get_dmodel(tensors)
end
x = decref["l"]
decode(x, dmodel)
