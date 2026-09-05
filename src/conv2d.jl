# I found a nice explanation for how to calculate convolution efficiently.
# https://github.com/alisaaalehi/convolution_as_multiplication
# Thank you for the author of the explanation PDF.

function conv2d_new_new(I, F, B)
    num_x, num_y = size(I)  # [x, y]
    @assert num_x == num_y
    a = num_x
    @assert size(F) == (3, 3)  # [ξ, η]
    
    FF₁ = @views [1 ≤ col - row + 2 ≤ 3 ? F[col - row + 2, 1] : 0 for row=1:a, col=1:a]
    FF₂ = @views [1 ≤ col - row + 2 ≤ 3 ? F[col - row + 2, 2] : 0 for row=1:a, col=1:a]
    FF₃ = @views [1 ≤ col - row + 2 ≤ 3 ? F[col - row + 2, 3] : 0 for row=1:a, col=1:a]

    FFs = [FF₁, FF₂, FF₃]
    z = fill(0, (a, a))

    FF = @views hcat([vcat([1 ≤ col - row + 2 ≤ 3 ? FFs[col - row + 2] : z for row=1:a]...) for col=1:a]...)

    II = vec(I)

    result = reshape(FF * II .+ B, (a, a))
end

conv2d_new_new([1 4 7; 2 5 8; 3 6 9], [10 40 70; 20 50 80; 30 60 90], 1) |> println
conv2d_new_new([1 4 7; 2 5 8; 3 6 9], [0 0 0; 0 1 0; 0 0 0], 0) |> println
