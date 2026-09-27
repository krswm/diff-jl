# I learned that there's "im2col"-based matmul-based 2D convolution algorithm.
# It's similar to what I did on "conv2d.jl" but modify `I` instead of `F`

# Thank you for teaching me: (websites)
# - https://github.com/alisaaalehi/convolution_as_multiplication
# - https://numb3r33.github.io/experiments/convolution/math/deeplearning/2023/12/23/im2col.html
# - https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/

using NNlib: conv

function tshow(x)
    show(IOContext(stdout, :limit => true), "text/plain", x)
    println()
end

# Convolution kernel
F = [
    11.0 14.0 17.0
    12.0 15.0 18.0
    13.0 16.0 19.0;;;

    21.0 24.0 27.0
    22.0 25.0 28.0
    23.0 26.0 29.0;;;;

    31.0 34.0 37.0
    32.0 35.0 38.0
    33.0 36.0 39.0;;;

    41.0 44.0 47.0
    42.0 45.0 48.0
    43.0 46.0 49.0;;;;
]  # [ξ, η, i, o]

F |> tshow
num_ξ, num_η, num_i, num_o = size(F)

# Convolution input
I = [
    110.0 140.0 170.0
    120.0 150.0 180.0
    130.0 160.0 190.0;;;

    210.0 240.0 270.0
    220.0 250.0 280.0
    230.0 260.0 290.0;;;;

    310.0 340.0 370.0
    320.0 350.0 380.0
    330.0 360.0 390.0;;;

    410.0 440.0 470.0
    420.0 450.0 480.0
    430.0 460.0 490.0;;;;
]  # [x, y, i, n]

num_x, num_y, _, num_n = size(I)

I |> tshow

O_ = conv(I, F, stride = 1, pad = 1, flipped = true)
O_ |> tshow

num_p = num_ξ * num_η * num_i

FF = reshape(F, num_p, num_o)

FF |> tshow

num_q = num_x * num_y * num_n

# My own `im2col` clone
colmaj = Vector{eltype(I)}()
for n = 1:num_n
    for y = 1:num_y
        for x = 1:num_x
            for i = 1:num_i
                for Δy = -1:1  # Corresponds to η
                    for Δx = -1:1  # Corresponds to ξ
                        xx = x + Δx
                        yy = y + Δy
                        value = if 1 ≤ xx ≤ num_x && 1 ≤ yy ≤ num_y
                            I[xx, yy, i, n]
                        else
                            zero(eltype(I))
                        end
                        push!(colmaj, value)
                    end
                end
            end
        end
    end
end
II = reshape(colmaj, num_p, num_q)

II |> tshow

# FF [p, o]
# II [p, q]
# OO [o, q]

OO = transpose(FF) * II

OO |> tshow

O = permutedims(reshape(OO, num_o, num_x, num_y, num_n), (2, 3, 1, 4))  # [x, y, o, n]

O |> tshow

(O == O_) |> println
# Identical!
