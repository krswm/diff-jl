# I found a nice explanation for how to calculate convolution efficiently.
# https://github.com/alisaaalehi/convolution_as_multiplication
# Thank you for the author of the explanation PDF.

using LinearAlgebra
using SparseArrays

function tshow(x)
    println(size(x))
    show(IOContext(stdout, :limit => true), "text/plain", x)
    println()
end

function conv2d_new_new(I, F, B)
    num_x, num_y = size(I)  # [x, y]
    @assert num_x == num_y
    a = num_x
    @assert size(F) == (3, 3)  # [ξ, η]

    @time begin
    
        FF₁_sp = spdiagm(-1 => fill(F[1, 1], a - 1), 0 => fill(F[2, 1], a), 1 => fill(F[3, 1], a - 1))
        FF₂_sp = spdiagm(-1 => fill(F[1, 2], a - 1), 0 => fill(F[2, 2], a), 1 => fill(F[3, 2], a - 1))
        FF₃_sp = spdiagm(-1 => fill(F[1, 3], a - 1), 0 => fill(F[2, 3], a), 1 => fill(F[3, 3], a - 1))


        # FF_sp = spdiagm(-1 => fill(FF₁_sp, a - 1), 0 => fill(FF₂_sp, a), 1 => fill(FF₃_sp, a - 1))
        FFs_sp = [FF₁_sp, FF₂_sp, FF₃_sp]
        z_sp = spzeros(Int, a, a)
        FF_sp  = sparse_hcat(
            [sparse_vcat(
                [1 ≤ col - row + 2 ≤ 3 ? FFs_sp[col - row + 2] : z_sp for row=1:a]
            ...) for col=1:a]
        ...)
        #=
        pairs = Dict(
            b => (
                b == -1 ? FF₁_sp :
                b ==  0 ? FF₂_sp :
                b ==  1 ? FF₃_sp :
                spzeros(Int, a, a)
            ) for b=(-a + 1):(a - 1)
        )
        FF_sp = diagm(pairs...)
        =#
        FF_sp |> tshow
    end

    II = vec(I)

    @time begin
        FF_sp = sparse(
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
            ),
            vcat(
                ((1:a-1) .+ (r-1)*a for r=1:a-1)...,
                ((1:a-1) .+ (r-1)*a for r=1:a  )...,
                ((1:a-1) .+ (r-1)*a for r=2:a  )...,

                ((1:a  ) .+ (r-1)*a for r=1:a-1)...,
                ((1:a  ) .+ (r-1)*a for r=1:a  )...,
                ((1:a  ) .+ (r-1)*a for r=2:a  )...,

                ((2:a  ) .+ (r-1)*a for r=1:a-1)...,
                ((2:a  ) .+ (r-1)*a for r=1:a  )...,
                ((2:a  ) .+ (r-1)*a for r=2:a  )...,
            ),
            vcat(
                fill(F[1, 1], (a-1)*(a-1)),  # "10"
                fill(F[1, 2], a*(a-1)    ),  # "40"
                fill(F[1, 3], (a-1)*(a-1)),  # "70"

                fill(F[2, 1], a*(a-1)    ),  # "20"
                fill(F[2, 2], a*a        ),  # "50"
                fill(F[2, 3], a*(a-1)    ),  # "80"

                fill(F[3, 1], (a-1)*(a-1)),  # "30"
                fill(F[3, 2], a*(a-1)    ),  # "60"
                fill(F[3, 3], (a-1)*(a-1)),  # "90"
            ),
        ) |> dropzeros!
        FF_sp |> tshow
    end

    II = vec(I)

    reshape(FF_sp * II .+ B, (a, a))
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

#=
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
=#
I = rand(Int, 512, 512)
# Transposing because I'll use [x, y], not [y, x] although visually it's diagonally flipped in the matrix form.

generate_pgm(I, ARGS[1])

O = conv2d_new_new(I, [1 1 2; 2 2 1; 1 2 1], 0)
generate_pgm(O, ARGS[2])

#=
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

conv2d_new_new(I, F, 0)
=#
