# I found a nice explanation for how to calculate convolution efficiently.
# https://github.com/alisaaalehi/convolution_as_multiplication

# Does it really work? Let me find out!

function conv2d_new(I, F)
    m₁, n₁ = size(I)
    m₂, n₂ = size(F)
    m = m₁ + m₂ - 1
    n = n₁ + n₂ - 1

    zpF = @views [1 ≤ y - 1 ≤ m₂ && 1 ≤ x ≤ n₂ ? F[y - 1, x] : 0 for y=1:m, x=1:n]

    # Toeplitz matrices
    F₀ = @views [1 ≤ y - x + 1 ≤ n ? zpF[3, y - x + 1] : 0 for y=1:n, x=1:n₁]
    F₁ = @views [1 ≤ y - x + 1 ≤ n ? zpF[2, y - x + 1] : 0 for y=1:n, x=1:n₁]
    F₂ = @views [1 ≤ y - x + 1 ≤ n ? zpF[1, y - x + 1] : 0 for y=1:n, x=1:n₁]

    db = hcat(vcat(F₀, F₁, F₂), vcat(fill(0, (n, n₁)), F₀, F₁))

    vI = vec(rotr90(I))

    rv = db * vI
    rotl90(reshape(rv, (n, m)))
end

x = [1 2 3; 4 5 6]  # [y, x]
w = [10 20; 30 40]  # [η, ξ]

output = conv2d_new(x, w)
output |> println

function conv2d_new(x, w)
    num_x, num_y = size(x)
    num_ξ, num_η = size(w)
    num_xₒ = num_x + num_ξ - 1
    num_yₒ = num_y + num_η - 1

    zpw = @views [1 ≤ x ≤ num_ξ && 1 ≤ y - 1 ≤ num_η ? w[x, y - 1] : 0 for x=1:num_xₒ, y=1:num_yₒ]  # [xₒ, yₒ]

    # Toeplitz matrices
    w₀ = @views [1 ≤ y - x + 1 ≤ num_xₒ ? zpw[y - x + 1, 3] : 0 for x=1:num_x, y=1:num_xₒ]  # [x, xₒ]
    w₁ = @views [1 ≤ y - x + 1 ≤ num_xₒ ? zpw[y - x + 1, 2] : 0 for x=1:num_x, y=1:num_xₒ]  # [x, xₒ]
    w₂ = @views [1 ≤ y - x + 1 ≤ num_xₒ ? zpw[y - x + 1, 1] : 0 for x=1:num_x, y=1:num_xₒ]  # [x, xₒ]

    db = vcat(hcat(w₀, w₁, w₂), hcat(fill(0, (num_x, num_xₒ)), w₀, w₁))

    vx = vcat(x[:, 2], x[:, 1])
    vx = insertdims(vx, dims=1)

    rv = vx * db
    reshape(rv, (num_xₒ, num_yₒ))
end

x = permutedims([1 2 3; 4 5 6], (2, 1))  # [y, x]
w = permutedims([10 20; 30 40], (2, 1))  # [η, ξ]

output = conv2d_new(x, w)
output |> println

# It works!

# conv2d is a commutative operation? I haven't realized that!

# Thank you for the author of the explanation PDF.
