using SparseArrays

using JSON
using SafeTensors

include("model.jl")
using .Model

function tshow(x)
    println(size(x))
    show(IOContext(stdout, :limit => true), "text/plain", x)
    println()
end

function conv2d_inner(x_vec, wc, bc, a, num_i)
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

function conv2d(x, wc, bc)
    # I found a nice explanation for how to calculate convolution efficiently.
    # https://github.com/alisaaalehi/convolution_as_multiplication
    # Thank you for the author of the explanation PDF.

    @assert ndims(x) == 3    
    num_x, num_y, num_i = size(x)
    @assert num_x == num_y  # My implementation supports only square image currently.
    a = num_x

    @assert ndims(wc) == 4
    @assert eltype(wc) == eltype(x)
    num_ξ, num_η, num_i_, num_o = size(wc)
    @assert num_ξ == num_η
    @assert num_i_ == num_i
    wc = eachslice(wc, dims=4)  # [o][ξ, η, i]

    @assert ndims(bc) == 1
    @assert eltype(bc) == eltype(x)
    num_o_, = size(bc)
    @assert num_o_ == num_o

    vec_x = vec(x)  # [αᵢ]

    result = conv2d_inner.(Ref(vec_x), wc, bc, a, num_i)  # [o][x, y]
    stack(result)
end

function decode(x, model)
    @assert size(x) == (1, 4, 64, 64)
    @assert eltype(x) == Float32
    x = reshape(x, (4, 64, 64))  # [o₀, y, x]
    x = permutedims(x, (3, 2, 1))  # [x, y, o₀]
    @assert size(x) == (64, 64, 4)
    @assert eltype(x) == Float32
    x ./= 0.18215f0
    @assert size(x) == (64, 64, 4)
    @assert eltype(x) == Float32
    x = conv2d(x, model.conv_pq.wc, model.conv_pq.bc)
    @assert size(x) == (64, 64, 4)
    @assert eltype(x) == Float32
    x |> tshow
end

decref = load_safetensors("../../../Downloads/decref.safetensors")
model = begin
    tensors = load_safetensors("../../../Downloads/sd/v1-5/model.safetensors")
    get_decoder_model(tensors)
end
x = decref["l"]
decode(x, model)
