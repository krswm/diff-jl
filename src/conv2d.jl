# I found a nice explanation for how to calculate convolution efficiently.
# https://github.com/alisaaalehi/convolution_as_multiplication
# Thank you for the author of the explanation PDF.

# I learnt that there's a FFT-based convolution algorithm.
# Maybe I'll implement it later, not now...
#
# My matmul-based implementation is not fast enough for 64x64x512ch image with 3x3x512ch kernel convolution...
# It's been already taking me a couple days just only for convolution...
# Let me cheat! --- That means, using an external NN library only for 2D convolutoion calculation.
#
# Flux.jl!
using Flux

using LinearAlgebra
using SparseArrays

function tshow(x)
    println(size(x))
    show(IOContext(stdout, :limit => true), "text/plain", x)
    println()
end

function get_OOi(Fi, Ii, a, rows, cols)
    vals = vcat(  # vals
        fill(Fi[1, 1], (a-1)*(a-1)),  #  "10"
        fill(Fi[1, 2], a*(a-1)    ),  #  "40"
        fill(Fi[1, 3], (a-1)*(a-1)),  #  "70"

        fill(Fi[2, 1], a*(a-1)    ),  #  "20"
        fill(Fi[2, 2], a*a        ),  #  "50"
        fill(Fi[2, 3], a*(a-1)    ),  #  "80"

        fill(Fi[3, 1], (a-1)*(a-1)),  #  "30"
        fill(Fi[3, 2], a*(a-1)    ),  #  "60"
        fill(Fi[3, 3], (a-1)*(a-1)),  #  "90"
    )
    FFi_sp = sparse(rows, cols, vals)
    IIi = vec(Ii)
    FFi_sp * IIi
end


function conv2d_new_new(I, F)
    num_x, num_y, num_i = size(I)  # [x, y, i]
    @assert num_x == num_y
    a = num_x
    num_ξ, num_η, num_i_ = size(F)  # [ξ, η, i]
    @assert num_ξ == num_η == 3

    II = vec(I)

    print("\x1b[31m")
    @time O = begin
        FF_sp = sparse(
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
        )
        reshape(FF_sp * II, (a, a))
    end
    O |> tshow
    print("\x1b[39m")

    print("\x1b[32m")
    @time O = begin
        F_ = eachslice(F, dims=3)  # [i][x, y]
        I_ = eachslice(I, dims=3)  # [i][x, y]

        rows = vcat(  # rows
            ((2:a  ) .+ (c-1)*a for c=2:a  )...,
            ((2:a  ) .+ (c-1)*a for c=1:a  )...,
            ((2:a  ) .+ (c-1)*a for c=1:a-1)...,

            ((1:a  ) .+ (c-1)*a for c=2:a  )...,
            ((1:a  ) .+ (c-1)*a for c=1:a  )...,
            ((1:a  ) .+ (c-1)*a for c=1:a-1)...,

            ((1:a-1) .+ (c-1)*a for c=2:a  )...,
            ((1:a-1) .+ (c-1)*a for c=1:a  )...,
            ((1:a-1) .+ (c-1)*a for c=1:a-1)...,
        )
        cols = vcat(  # cols
            ((1:a-1) .+ (r-1)*a for r=1:a-1)...,
            ((1:a-1) .+ (r-1)*a for r=1:a  )...,
            ((1:a-1) .+ (r-1)*a for r=2:a  )...,

            ((1:a  ) .+ (r-1)*a for r=1:a-1)...,
            ((1:a  ) .+ (r-1)*a for r=1:a  )...,
            ((1:a  ) .+ (r-1)*a for r=2:a  )...,

            ((2:a  ) .+ (r-1)*a for r=1:a-1)...,
            ((2:a  ) .+ (r-1)*a for r=1:a  )...,
            ((2:a  ) .+ (r-1)*a for r=2:a  )...,
        )
        OOi = get_OOi.(F_, I_, a, Ref(rows), Ref(cols))  # [i][xy]
        result = stack(OOi)  # [xy, i]
        result = sum(result, dims=2)  # [xy]
        reshape(result, (a, a))  # [x, y]
    end
    O |> tshow
    print("\x1b[39m")

    print("\x1b[33m")
    I_ = insertdims(I, dims=4)
    F_ = insertdims(F, dims=4)
    @time O = conv(I_, F_, stride=1, pad=1)  # Over x10 faster than mine!? Memory usage about 1/7!?
    O |> tshow
    print("\x1b[39m")

    #=
    @time FF_sp_noi = sparse(
        vcat(  # rows
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
        vcat(  # cols
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
        vcat(  # vals
            fill(F[1, 1], (a-1)*(a-1)),  #  "10"
            fill(F[1, 2], a*(a-1)    ),  #  "40"
            fill(F[1, 3], (a-1)*(a-1)),  #  "70"

            fill(F[2, 1], a*(a-1)    ),  #  "20"
            fill(F[2, 2], a*a        ),  #  "50"
            fill(F[2, 3], a*(a-1)    ),  #  "80"

            fill(F[3, 1], (a-1)*(a-1)),  #  "30"
            fill(F[3, 2], a*(a-1)    ),  #  "60"
            fill(F[3, 3], (a-1)*(a-1)),  #  "90"
        ),
        a*a,
        a*a,
    )
    FF_sp |> tshow

    reshape(FF_sp__ * II, (a, a)) |> tshow
    =#

    #=
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
    =#

    # incorrect
    #=
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
    =#

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

I = rand(64, 64, 512)
F = rand(3, 3, 512)

F |> tshow

conv2d_new_new(I, F)

println("----")

I = rand(64, 64, 512)
F = rand(3, 3, 512)

conv2d_new_new(I, F)
