# I learned that there's "im2col"-based matmul-based 2D convolution algorithm.
# It's similar to what I did on "conv2d.jl" but modify `I` instead of `F`

# Thank you for teaching me: (websites)
# - https://github.com/alisaaalehi/convolution_as_multiplication
# - https://numb3r33.github.io/experiments/convolution/math/deeplearning/2023/12/23/im2col.html
# - https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/

# using NNlib: conv

function tshow(x)
    show(IOContext(stdout, :limit => true), "text/plain", x)
    println()
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

num_ξ = 3
num_η = 3
num_x = 64
num_y = 64
num_i = 320
num_o = 320
num_n = 2
stride = 2
pad = num_ξ ÷ 2

# Convolution kernel
# F = [
#     11.0 14.0 17.0
#     12.0 15.0 18.0
#     13.0 16.0 19.0;;;
# 
#     21.0 24.0 27.0
#     22.0 25.0 28.0
#     23.0 26.0 29.0;;;;
# 
#     31.0 34.0 37.0
#     32.0 35.0 38.0
#     33.0 36.0 39.0;;;
# 
#     41.0 44.0 47.0
#     42.0 45.0 48.0
#     43.0 46.0 49.0;;;;
# ]  # [ξ, η, i, o]

# Convolution input
# I = [
#     110.0 140.0 170.0
#     120.0 150.0 180.0
#     130.0 160.0 190.0;;;
# 
#     210.0 240.0 270.0
#     220.0 250.0 280.0
#     230.0 260.0 290.0;;;;
# 
#     310.0 340.0 370.0
#     320.0 350.0 380.0
#     330.0 360.0 390.0;;;
# 
#     410.0 440.0 470.0
#     420.0 450.0 480.0
#     430.0 460.0 490.0;;;;
# ]  # [x, y, i, n]

F = rand(num_ξ, num_η, num_i, num_o)
I = rand(num_x, num_y, num_i, num_n)
@time O = my_conv(I, F, stride)
@time O_ = conv(I, F, stride = stride, pad = pad, flipped = true)
(O == O_) |> println

F = rand(num_ξ, num_η, num_i, num_o)
I = rand(num_x, num_y, num_i, num_n)
@time O = my_conv(I, F, stride)
@time O_ = conv(I, F, stride = stride, pad = pad, flipped = true)
(O == O_) |> println
