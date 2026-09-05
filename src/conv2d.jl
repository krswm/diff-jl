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
    output = rotl90(reshape(rv, (n, m)))
end

I = [1 2 3; 4 5 6]
F = [10 20; 30 40]

conv2d_new(I, F)
output |> println

# It works!

# conv2d is a commutative operation? I haven't realized that!

# Thank you for the author of the explanation PDF.
