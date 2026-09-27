using DelimitedFiles
using SparseArrays
using Statistics

using JSON
using SafeTensors

function tshow(x, color)
    print("\x1b[$(color)m")
    println(size(x))
    show(IOContext(stdout, :limit => true), "text/plain", x)
    println()
    print("\x1b[39m")
end

function my_conv(I, F, stride)
    num_ξ, num_η, num_i, num_o = size(F)
    num_x, num_y, _, num_n = size(I)
    num_p = num_ξ * num_η * num_i
    num_X = num_x ÷ stride
    num_Y = num_y ÷ stride
    num_Z = num_X * num_Y

    @assert num_ξ == num_η
    @assert num_ξ % 2 == 1
    pad = num_ξ ÷ 2

    # II: My own `im2col` clone
    II = [
        begin
            xx = (X - 1) * stride + ξ - pad
            yy = (Y - 1) * stride + η - pad
            value = if 1 ≤ xx ≤ num_x && 1 ≤ yy ≤ num_y
                I[xx, yy, i, n]
            else
                zero(eltype(I))
            end
        end for X = 1:num_X, Y = 1:num_Y, ξ = 1:num_ξ, η = 1:num_η, i = 1:num_i, n = 1:num_n
    ]  # [X, Y, ξ, η, i, n]
    II = reshape(II, num_Z, num_p, num_n)  # [Z, p, n]
    II = eachslice(II, dims=3)  # [n][Z, p]

    FF = reshape(F, num_p, num_o)  # [p, o]

    OO = II .* Ref(FF)  # [n][Z, o]
    OO = stack(OO)  # [Z, o, n]
    OO = reshape(OO, num_X, num_Y, num_o, num_n)  # [X, Y, o, n]

    OO
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
    y_var = var(y, dims=(1, 2, 3), corrected=false)  # [x, y, j, g, n] (x=1, y=1, j=1 only)
    y_var = [y_var[1, 1, 1, (o - 1) ÷ num_j + 1, n] for o=1:num_o, n=1:num_n]  # [o, n]
    x = eachslice(x, dims=(3, 4))  # [o, n][x, y]
    x = [norm_inner(x[o, n], g[o], t[o], y_mean[o, n], y_var[o, n]) for o=1:num_o, n=1:num_n]  # [o, n][x, y]
    stack(x)  # [x, y, o, n]
end

function layernorm(x, g, t)
    # x [x, y, o, n]
    # g [o]
    # t [o]
    num_x, num_y, num_o, num_n = size(x)

    x_mean = mean(x, dims=(1, 2, 3))  # [x, y, o, n] (x=1, y=1, o=1 only)
    x_mean = reshape(x_mean, (num_n,))  # [o, n]
    x_var = var(x, dims=(1, 2, 3), corrected=false)  # [x, y, o, n] (x=1, y=1, o=1 only)
    x_var = reshape(x_var, (num_n,))  # [o, n]
    x = eachslice(x, dims=(3, 4))  # [o, n][x, y]
    x = [norm_inner(x[o, n], g[o], t[o], x_mean[n], x_var[n]) for o=1:num_o, n=1:num_n]  # [o, n][x, y]
    stack(x)  # [x, y, o, n]
end

silu(x) = x / (exp(-x) + 1)

function calc_drblock(x, drblock)
    y = x  # [x, y, o, n]
    x = groupnorm(x, drblock.g1, drblock.t1)  # [x, y, o, n]
    x = silu.(x)  # [x, y, o, n]
    x = my_conv(x, drblock.wc1, 1) .+ drblock.bc1
    x = groupnorm(x, drblock.g2, drblock.t2)  # [x, y, o, n]
    x = silu.(x)  # [x, y, o, n]
    x = my_conv(x, drblock.wc2, 1) .+ drblock.bc2
    y + x
end

function calc_drcblock(x, drcblock)
    y = x  # [x, y, o, n]
    x = groupnorm(x, drcblock.g1, drcblock.t1)  # [x, y, o, n]
    x = silu.(x)  # [x, y, o, n]
    x = my_conv(x, drcblock.wc1, 1) .+ drcblock.bc1
    x = groupnorm(x, drcblock.g2, drcblock.t2)  # [x, y, o, n]
    x = silu.(x)  # [x, y, o, n]
    x = my_conv(x, drcblock.wc2, 1) .+ drcblock.bc2
    x + (my_conv(y, drcblock.wc3, 1) .+ drcblock.bc3)
end

function softmax(x; dims)
    # Numerically stable softmax

    #                     exp(xᵢ)        exp(xᵢ) exp(-xₘₐₓ)        exp(xᵢ - xₘₐₓ) 
    # [softmax(x)]ᵢ ≡ ------------ = ----------------------- = -------------------
    #                  ∑ⱼ exp(xⱼ)     ∑ⱼ exp(xⱼ) exp(-xₘₐₓ)     ∑ⱼ exp(xⱼ - xₘₐₓ) 
    #
    #                                                           ↑ this algorithm
    #
    # The number inside `exp` is guaranteed to be ≤0 thus stable.
    
    # x [a, b]

    numerator = exp.(x .- maximum(x, dims=dims))  # [a, b]
    denominator = sum(numerator, dims=dims)  # [a, b]
    numerator ./ denominator  # [a, b]
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
    #=
    @assert size(x) == (1, 4, 64, 64)  # [n, o, y, x]
    x = permutedims(x, (4, 3, 2, 1))  # [x, y, o, n]
    =#
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 4, 1)
    x ./= 0.18215f0
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 4, 1)

    @time x = my_conv(x, dmodel.dconv_pq.wc, 1) .+ dmodel.dconv_pq.bc
    @assert eltype(x) == Float32
    @assert size(x) == (64, 64, 4, 1)
    @time x = my_conv(x, dmodel.dconv_in.wc, 1) .+ dmodel.dconv_in.bc
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
    @time x = my_conv(x, dmodel.dconv_3.wc, 1) .+ dmodel.dconv_3.bc
    @assert eltype(x) == Float32
    @assert size(x) == (128, 128, 512, 1)

    @time x = calc_drblock(x, dmodel.drblock_20)
    @time x = calc_drblock(x, dmodel.drblock_21)
    @time x = calc_drblock(x, dmodel.drblock_22)
    @time x = upsample(x)
    @time x = my_conv(x, dmodel.dconv_2.wc, 1) .+ dmodel.dconv_2.bc

    @time x = calc_drcblock(x, dmodel.drcblock_10)
    @time x = calc_drblock(x, dmodel.drblock_11)
    @time x = calc_drblock(x, dmodel.drblock_12)
    @time x = upsample(x)
    @time x = my_conv(x, dmodel.dconv_1.wc, 1) .+ dmodel.dconv_1.bc

    @time x = calc_drcblock(x, dmodel.drcblock_00)
    @time x = calc_drblock(x, dmodel.drblock_01)
    @time x = calc_drblock(x, dmodel.drblock_02)

    x = groupnorm(x, dmodel.dgn.g, dmodel.dgn.t)
    x = silu.(x)
    @time x = my_conv(x, dmodel.dconv_out.wc, 1) .+ dmodel.dconv_out.bc

    # Expected result!
    # Thank you Flux.jl for providing me a fast 2D convolution implementation.
    
    x
end

function calc_frblock(x, f, frblock)
    y = x  # [x, y, o, n]
    x = groupnorm(x, frblock.g1, frblock.t1)  # [x, y, o, n]
    x = silu.(x)  # [x, y, o, n]
    x = my_conv(x, frblock.wc1, 1) .+ frblock.bc1
    f = silu.(f)  # [f₃]
    f = frblock.w * f .+ frblock.b  # [o]
    f = insertdims(f, dims=(1, 2, 4))  # [x, y, o, n]
    x .+= f  # [x, y, o, n]
    x = groupnorm(x, frblock.g2, frblock.t2)  # [x, y, o, n]
    x = silu.(x)  # [x, y, o, n]
    x = my_conv(x, frblock.wc2, 1) .+ frblock.bc2
    x + y
end

function calc_frcblock(x, f, frcblock)
    y = x  # [x, y, o, n]
    x = groupnorm(x, frcblock.g1, frcblock.t1)  # [x, y, o, n]
    x = silu.(x)  # [x, y, o, n]
    x = my_conv(x, frcblock.wc1, 1) .+ frcblock.bc1
    f = silu.(f)  # [f₃]
    f = frcblock.w * f .+ frcblock.b  # [o]
    f = insertdims(f, dims=(1, 2, 4))  # [x, y, o, n]
    x .+= f  # [x, y, o, n]
    x = groupnorm(x, frcblock.g2, frcblock.t2)  # [x, y, o, n]
    x = silu.(x)  # [x, y, o, n]
    x = my_conv(x, frcblock.wc2, 1) .+ frcblock.bc2
    x + (my_conv(y, frcblock.wc3, 1) .+ frcblock.bc3)
end

function layernorm(x, g, t)
    # x [xy, o]
    x_mean = mean(x, dims=2)  # [xy, 1]
    x_var = var(x, dims=2, corrected=false)  # [xy, 1]

    # g [o]
    g = insertdims(g, dims=1)  # [1, o]
    # t [o]
    t = insertdims(t, dims=1)  # [1, o]

    g .* (x .- x_mean) ./ .√(x_var .+ 1f-5) .+ t  # [xy, o]
end

function cross_attention(x, y, w1q, w1k, w1v, w2, b2, num_H)
    # x   [Dq₁, Sq ]
    # y   [Dkv, Skv]
    # w1q [Dq₂, Dq₁]
    # w1k [Dq₂, Dkv]
    # w1v [Dq₃, Dkv]
    # w2  [Dq₄, Dq]
    # b2  [Dq₄]
    b2 = insertdims(b2, dims=2)  # [Dq₄, 1]
    num_Dq, num_Sq = size(x)
    num_Dkv, num_Skv = size(y)
    num_I = num_Dq ÷ num_H

    q = w1q * x  # [Dq₂, Sq ]
    k = w1k * y  # [Dq₂, Skv]
    v = w1v * y  # [Dq₃, Skv]

    q = reshape(q, num_I, num_H, num_Sq)   # [I, H, Sq ]
    k = reshape(k, num_I, num_H, num_Skv)  # [I, H, Skv]
    v = reshape(v, num_I, num_H, num_Skv)  # [I, H, Skv]

    q = permutedims(q, (1, 3, 2))  # [I, Sq,  H]
    k = permutedims(k, (1, 3, 2))  # [I, Skv, H]
    v = permutedims(v, (1, 3, 2))  # [I, Skv, H]

    q = eachslice(q, dims=3)  # [H][I, Sq ]
    k = eachslice(k, dims=3)  # [H][I, Skv]
    v = eachslice(v, dims=3)  # [H][I, Skv]

    d = √convert(eltype(x), num_I)
    x = @. v * softmax(transpose(k) * q / d, dims=1)  # @. [H][I, Skv] * [H][Skv, Sq] -> [H][I, Sq]
    x = stack(x)                                      # [I, Sq, H]
    x = permutedims(x, (1, 3, 2))                     # [I, H, Sq]
    x = reshape(x, num_Dq, num_Sq)                    # [Dq, Sq]
    w2 * x .+ b2                                      # [Dq₄, Sq]
end

ugelu(v) = (tanh((v ^ 3 * 0.044715f0 + v) * sqrt(2.0f0 / pi)) + 1.0f0) * v * 0.5f0

function calc_fablock_4(x, w1, b1, w2, b2)
    b1 = insertdims(b1, dims=2)
    b2 = insertdims(b2, dims=2)
    x = w1 * x .+ b1  # [A, B]
    num_A, num_B = size(x)
    x = reshape(x, num_A ÷ 2, 2, num_B)
    x, g = eachslice(x, dims=2)
    x = x .* ugelu.(g)
    x = w2 * x .+ b2  # [A, B]
end


function calc_fablock(x, c, fablock)
    # x [x, y, o, n]
    num_x, num_y, num_o, num_n = size(x)
    num_xy = num_x * num_y

    y = x  # [x, y, o, n]

    x = groupnorm(x, fablock.g1, fablock.t1)  # [x, y, o, n]
    x = my_conv(x, fablock.wc1, 1) .+ fablock.bc1  # [x, y, o, n]

    x = reshape(x, (num_xy, num_o, num_n))  # [xy, o, n]
    x = eachslice(x, dims=3)  # [n][xy, o]

    z = x  # [n][xy, o]
    z = layernorm.(x, Ref(fablock.g2), Ref(fablock.t2))  # [n][xy, o]
    z = self_attention.(z, Ref(fablock.w21), Ref(fablock.b21), Ref(fablock.w22), Ref(fablock.b22), 8)  # [n][xy, o]
    x += z  # [n][xy, o]

    z = x  # [n][xy, o]
    z = layernorm.(x, Ref(fablock.g3), Ref(fablock.t3))  # [n][xy, o]
    z = transpose.(z)  # [n][o, xy]
    c = eachslice(c, dims=3)  # [n][Dkv, Skv]
    z = cross_attention.(z, c, Ref(fablock.w31q), Ref(fablock.w31k), Ref(fablock.w31v), Ref(fablock.w32), Ref(fablock.b32), 8)  # [n, o, xy]
    z = transpose.(z)  # [n][xy, o]
    x += z

    z = x  # [n][xy, o]
    z = layernorm.(z, Ref(fablock.g4), Ref(fablock.t4))  # [n][xy, o]
    z = transpose.(z)  # [n][o, xy]
    z = calc_fablock_4.(z, Ref(fablock.w41), Ref(fablock.b41), Ref(fablock.w42), Ref(fablock.b42))  # [n][xy, o]
    z = transpose.(z)
    x += z  # [n][xy, o]

    x = stack(x)  # [xy, o, n]
    x = reshape(x, num_x, num_y, num_o, num_n)  # [x, y, o, n]
    x = my_conv(x, fablock.wc4, 1) .+ fablock.bc4  # [x, y, o, n]
    y += x
    y
end

function denoise(x, c, t, prev_t, fmodel)
    println("==== t = $t ====")

    y = x

    f = t .* 10000 .^ (0.0f0:(-1.0f0/160):(-159.0f0/160))  # [f₁]
    f = vcat(cos.(f), sin.(f))  # [f₁]

    f = fmodel.time_w1 * f + fmodel.time_b1  # [f₂]
    f = silu.(f)  # [f₂]
    f = fmodel.time_w2 * f + fmodel.time_b2  # [f₃]

    x = cat(x, x, dims = 4)  # [x, y, o, n]

    print("0.0 ")
    @time x = my_conv(x, fmodel.fconv_i0.wc, 1) .+ fmodel.fconv_i0.bc
    s0 = x
    print("1.0 ")
    @time x = calc_frblock(x, f, fmodel.frblock_i1)
    print("1.1 ")
    @time x = calc_fablock(x, c, fmodel.fablock_i1)
    s1 = x
    print("2.0 ")
    @time x = calc_frblock(x, f, fmodel.frblock_i2)
    print("2.1 ")
    @time x = calc_fablock(x, c, fmodel.fablock_i2)
    s2 = x

    print("3.0 ")
    @time x = my_conv(x, fmodel.fconv_i3.wc, 2) .+ fmodel.fconv_i3.bc
    s3 = x
    print("4.0 ")
    @time x = calc_frcblock(x, f, fmodel.frcblock_i4)
    print("4.1 ")
    @time x = calc_fablock(x, c, fmodel.fablock_i4)
    s4 = x
    print("5.0 ")
    @time x = calc_frblock(x, f, fmodel.frblock_i5)
    print("5.1 ")
    @time x = calc_fablock(x, c, fmodel.fablock_i5)
    s5 = x

    print("6.0 ")
    @time x = my_conv(x, fmodel.fconv_i6.wc, 2) .+ fmodel.fconv_i6.bc
    s6 = x
    print("7.0 ")
    @time x = calc_frcblock(x, f, fmodel.frcblock_i7)
    print("7.1 ")
    @time x = calc_fablock(x, c, fmodel.fablock_i7)
    s7 = x
    print("8.0 ")
    @time x = calc_frblock(x, f, fmodel.frblock_i8)
    print("8.1 ")
    @time x = calc_fablock(x, c, fmodel.fablock_i8)
    s8 = x

    print("9.0 ")
    @time x = my_conv(x, fmodel.fconv_i9.wc, 2) .+ fmodel.fconv_i9.bc
    s9 = x
    print("10.0 ")
    @time x = calc_frblock(x, f, fmodel.frblock_i10)
    s10 = x
    print("11.0 ")
    @time x = calc_frblock(x, f, fmodel.frblock_i11)
    s11 = x

    print("m0 ")
    @time x = calc_frblock(x, f, fmodel.frblock_m0)
    print("m1 ")
    @time x = calc_fablock(x, c, fmodel.fablock_m1)
    print("m2 ")
    @time x = calc_frblock(x, f, fmodel.frblock_m2)

    x = cat(x, s11; dims=3)
    print("d0.0 ")
    @time x = calc_frcblock(x, f, fmodel.frcblock_o0)
    x = cat(x, s10; dims=3)
    print("d1.0 ")
    @time x = calc_frcblock(x, f, fmodel.frcblock_o1)
    x = cat(x, s9; dims=3)
    print("d2.0 ")
    @time x = calc_frcblock(x, f, fmodel.frcblock_o2)
    print("d2.1 ")
    x = upsample(x)
    @time x = my_conv(x, fmodel.fconv_o2.wc, 1) .+ fmodel.fconv_o2.bc

    x = cat(x, s8; dims=3)
    print("d3.0 ")
    @time x = calc_frcblock(x, f, fmodel.frcblock_o3)
    print("d3.1 ")
    @time x = calc_fablock(x, c, fmodel.fablock_o3)
    x = cat(x, s7; dims=3)
    print("d4.0 ")
    @time x = calc_frcblock(x, f, fmodel.frcblock_o4)
    print("d4.1 ")
    @time x = calc_fablock(x, c, fmodel.fablock_o4)
    x = cat(x, s6; dims=3)
    print("d5.0 ")
    @time x = calc_frcblock(x, f, fmodel.frcblock_o5)
    print("d5.1 ")
    @time x = calc_fablock(x, c, fmodel.fablock_o5)
    print("d5.2 ")
    x = upsample(x)
    @time x = my_conv(x, fmodel.fconv_o5.wc, 1) .+ fmodel.fconv_o5.bc

    x = cat(x, s5; dims=3)
    print("d6.0 ")
    @time x = calc_frcblock(x, f, fmodel.frcblock_o6)
    print("d6.1 ")
    @time x = calc_fablock(x, c, fmodel.fablock_o6)
    x = cat(x, s4; dims=3)
    print("d7.0 ")
    @time x = calc_frcblock(x, f, fmodel.frcblock_o7)
    print("d7.1 ")
    @time x = calc_fablock(x, c, fmodel.fablock_o7)
    x = cat(x, s3; dims=3)
    print("d8.0 ")
    @time x = calc_frcblock(x, f, fmodel.frcblock_o8)
    print("d8.1 ")
    @time x = calc_fablock(x, c, fmodel.fablock_o8)
    print("d8.2 ")
    x = upsample(x)
    @time x = my_conv(x, fmodel.fconv_o8.wc, 1) .+ fmodel.fconv_o8.bc

    x = cat(x, s2; dims=3)
    print("d9.0 ")
    @time x = calc_frcblock(x, f, fmodel.frcblock_o9)
    print("d9.1 ")
    @time x = calc_fablock(x, c, fmodel.fablock_o9)
    x = cat(x, s1; dims=3)
    print("d10.0 ")
    @time x = calc_frcblock(x, f, fmodel.frcblock_o10)
    print("d10.1 ")
    @time x = calc_fablock(x, c, fmodel.fablock_o10)
    x = cat(x, s0; dims=3)
    print("d11.0 ")
    @time x = calc_frcblock(x, f, fmodel.frcblock_o11)
    print("d11.1 ")
    @time x = calc_fablock(x, c, fmodel.fablock_o11)

    x = groupnorm(x, fmodel.g_final, fmodel.t_final)
    x = silu.(x)
    x = my_conv(x, fmodel.wc_final, 1) .+ fmodel.bc_final  # [x, y, o, n]
    x_positive, x_negative = eachslice(x; dims=4)  # [x, y, o], [x, y, o]
    config_scale = 8
    x = config_scale .* (x_positive - x_negative) .+ x_negative  # [x, y, o]
    x = insertdims(x, dims=4)  # [x, y, o, 1]
    # y [x, y, o, 1]
    x = ddpm_step(t, prev_t, y, x)  # [x, y, o, 1]

    x    
end

function diffuse(c, fmodel)
    #=
    rand42 = load_safetensors("../../../Downloads/rand42.safetensors")
    x = permutedims(rand42["l"], (4, 3, 2, 1))  # [x, y, o, n]
    =#
    x = randn(Float32, 64, 64, 4, 1)  # [x, y, o, n]
    # A different image of a cat with a hat generated when I change this to randn!
    # It's a good sign.

    x = denoise(x, c, 900,  800, fmodel)
    x = denoise(x, c, 800,  700, fmodel)
    x = denoise(x, c, 700,  600, fmodel)
    x = denoise(x, c, 600,  500, fmodel)
    x = denoise(x, c, 500,  400, fmodel)
    x = denoise(x, c, 400,  300, fmodel)
    x = denoise(x, c, 300,  200, fmodel)
    x = denoise(x, c, 200,  100, fmodel)
    x = denoise(x, c, 100,    0, fmodel)
    x = denoise(x, c,   0, -100, fmodel)
    # Expected result!
    x
end

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
        println(file)
    end
end

function generate_image(c, model_path, output_path)
    fmodel = begin
        tensors = load_safetensors("$model_path/model.safetensors")
        get_fmodel(tensors)
    end

    x = diffuse(c, fmodel)

    println("==== Decoding latent space -> image (RGB) space ====")
    dmodel = begin
        tensors = load_safetensors("$model_path/model.safetensors")
        get_dmodel(tensors)
    end
    x = decode(x, dmodel)

    x = clamp.(floor.((x .+ 1) * 128), UInt8)  # [x, y, rgb, n]

    generate_ppm_image(x, output_path)
end

