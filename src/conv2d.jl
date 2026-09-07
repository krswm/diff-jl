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

function conv2d_new_new(I, F)
    num_x, num_y, num_i = size(I)  # [x, y, i]
    @assert num_x == num_y
    a = num_x
    num_ξ, num_η, num_i_ = size(F)  # [ξ, η, i]
    @assert num_ξ == num_η == 3

    @time FF_sp = sparse(
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
                    fill(F[1, 1, i], (a-1)*(a-1)),  #  "10"
                    fill(F[1, 2, i], a*(a-1)    ),  #  "40"
                    fill(F[1, 3, i], (a-1)*(a-1)),  #  "70"

                    fill(F[2, 1, i], a*(a-1)    ),  #  "20"
                    fill(F[2, 2, i], a*a        ),  #  "50"
                    fill(F[2, 3, i], a*(a-1)    ),  #  "80"

                    fill(F[3, 1, i], (a-1)*(a-1)),  #  "30"
                    fill(F[3, 2, i], a*(a-1)    ),  #  "60"
                    fill(F[3, 3, i], (a-1)*(a-1)),  #  "90"
                ) for i=1:num_i
            )...
        ),
        a*a,
        a*a*num_i,
    ) |> dropzeros!
    FF_sp |> tshow

    II = vec(I)

    z = zero(eltype(F))
    @time FF_sp_ = sparse_hcat(
        (
            spdiagm(
                -a-1 => collect(Iterators.take(Iterators.cycle([F[1, 1, i], F[1, 1, i], z         ]), a*(a-1)-1)),
                -a   => collect(Iterators.take(Iterators.cycle([F[2, 1, i], F[2, 1, i], F[2, 1, i]]), a*(a-1)  )),
                -a+1 => collect(Iterators.take(Iterators.cycle([z,          F[3, 1, i], F[3, 1, i]]), a*(a-1)+1)),

                  -1 => collect(Iterators.take(Iterators.cycle([F[1, 2, i], F[1, 2, i], z         ]), a*a    -1)),
                   0 => collect(Iterators.take(Iterators.cycle([F[2, 2, i], F[2, 2, i], F[2, 2, i]]), a*a      )),
                   1 => collect(Iterators.take(Iterators.cycle([F[3, 2, i], F[3, 2, i], z         ]), a*a    -1)),

                 a-1 => collect(Iterators.take(Iterators.cycle([z,          F[1, 3, i], F[1, 3, i]]), a*(a-1)+1)),
                 a   => collect(Iterators.take(Iterators.cycle([F[2, 3, i], F[2, 3, i], F[2, 3, i]]), a*(a-1)  )),
                 a+1 => collect(Iterators.take(Iterators.cycle([F[3, 3, i], F[3, 3, i], z         ]), a*(a-1)-1)),
            ) |> dropzeros! for i=1:num_i
        )...
    )
    FF_sp_ |> tshow

    # incorrect
    @time FF_sp__ = spdiagm(
        a*a,
        a*a*num_i,
        merge(
            (
                Dict(
                    -a-1 + (a*a*(i-1)) => collect(Iterators.take(Iterators.cycle([F[1, 1, i], F[1, 1, i], z         ]), a*(a-1)-1)),
                    -a   + (a*a*(i-1)) => collect(Iterators.take(Iterators.cycle([F[2, 1, i], F[2, 1, i], F[2, 1, i]]), a*(a-1)  )),
                    -a+1 + (a*a*(i-1)) => collect(Iterators.take(Iterators.cycle([z,          F[3, 1, i], F[3, 1, i]]), a*(a-1)+1)),

                      -1 + (a*a*(i-1)) => collect(Iterators.take(Iterators.cycle([F[1, 2, i], F[1, 2, i], z         ]), a*a    -1)),
                       0 + (a*a*(i-1)) => collect(Iterators.take(Iterators.cycle([F[2, 2, i], F[2, 2, i], F[2, 2, i]]), a*a      )),
                       1 + (a*a*(i-1)) => collect(Iterators.take(Iterators.cycle([F[3, 2, i], F[3, 2, i], z         ]), a*a    -1)),

                     a-1 + (a*a*(i-1)) => collect(Iterators.take(Iterators.cycle([z,          F[1, 3, i], F[1, 3, i]]), a*(a-1)+1)),
                     a   + (a*a*(i-1)) => collect(Iterators.take(Iterators.cycle([F[2, 3, i], F[2, 3, i], F[2, 3, i]]), a*(a-1)  )),
                     a+1 + (a*a*(i-1)) => collect(Iterators.take(Iterators.cycle([F[3, 3, i], F[3, 3, i], z         ]), a*(a-1)-1)),
                ) for i=1:num_i
            )...
        )...
    ) |> dropzeros!
    FF_sp__ |> tshow

    reshape(FF_sp * II, (a, a)) |> tshow
    reshape(FF_sp__ * II, (a, a)) |> tshow
end

#=
using DelimitedFiles

function generate_pgm(x, filename)
    # NetPGM! One of the simplest image formats.
    num_x, num_y = size(x)
    open(filename, "w") do file
        println(file, "P2", " ", num_x, " ", num_y, " ", 16)
        writedlm(file, transpose(x))
    end
end
=#

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
#=
I = rand(Int, 512, 512)
# Transposing because I'll use [x, y], not [y, x] although visually it's diagonally flipped in the matrix form.

generate_pgm(I, ARGS[1])

O = conv2d_new_new(I, [1 1 2; 2 2 1; 1 2 1], 0)
generate_pgm(O, ARGS[2])
=#

#=
I = [
     1  4  7
     2  5  8
     3  6  9;;;
    11 14 17
    12 15 18
    13 16 19;;;
    21 24 27
    22 25 28
    23 26 29;;;
]

F = [
     10  40  70
     20  50  80
     30  60  90;;;
    110 140 170
    120 150 180
    130 160 190;;;
    210 240 270
    220 250 280
    230 260 290;;;
]
=#

#=
I = [
     1  4  7
     2  5  8
     3  6  9;;;
    11 14 17
    12 15 18
    13 16 19;;;
]

F = [
     10  40  70
     20  50  80
     30  60  90;;;
    210 240 270
    220 250 280
    230 260 290;;;
]

conv2d_new_new(I, F)
=#

I = rand(64, 64, 64)
F = rand(3, 3, 64)

F |> tshow

conv2d_new_new(I, F)

println("----")

I = rand(64, 64, 512)
F = rand(3, 3, 512)

conv2d_new_new(I, F)
