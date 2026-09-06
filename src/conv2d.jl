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

#=
function tshow(x)
    println(size(x))
    show(IOContext(stdout, :limit => true), "text/plain", x)
    println()
end

using DelimitedFiles

function generate_pgm(x, filename)
    # NetPGM! One of the simplest image formats.
    num_x, num_y = size(x)
    open(filename, "w") do file
        println(file, "P2", " ", num_x, " ", num_y, " ", 16)
        writedlm(file, transpose(x))
    end
end

I = [
    0 1 1 0 4 4 4 4
    0 1 1 0 0 0 4 4
    2 2 3 3 0 0 4 4
    2 2 3 3 0 0 4 4
    0 0 0 0 0 0 4 4
    4 4 0 0 0 0 4 4
    4 4 0 0 0 0 4 4
    0 4 4 4 4 4 4 0
] |> transpose
# Transposing because I'll use [x, y], not [y, x] although visually it's diagonally flipped in the matrix form.

generate_pgm(I, ARGS[1])

O = conv2d_new_new(I, [0 1 0; 1 8 1; 0 1 0], 0)
generate_pgm(O, ARGS[2])
=#

I = [
    1 4 7
    2 5 8
    3 6 9
]

F = [
    10 40 70
    20 50 80
    30 60 90
]

conv2d_new_new(I, F, 0) |> println
