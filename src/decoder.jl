using Statistics

using JSON
using SafeTensors

include("model.jl")
using .Model

function tshow(x)
    show(IOContext(stdout, :limit => true), "text/plain", x)
    println()
end

# TODO: Exactly the same as `conv2d` in main.jl. Unify with it.
function conv2d(wc, bc, latent)
    # Kernel size 3x3, padding 1

    N, Cin, H, W = size(latent)

    Cout, Cin_, HH, WW = size(wc)
    if !(Cin == Cin_ && HH == WW && HH % 2 == 1)
        print("$(size(latent))")
        print("$(size(wc))")
        print("$Cin $Cin_ $HH $WW")
        @assert false
    end

    kw = HH ÷ 2
    # kernelwidth = 0
    #
    # .....
    # .....
    # ..O..
    # .....
    # .....
    #
    # kernelwidth = 1
    #
    # .....
    # .OOO.
    # .OOO.
    # .OOO.
    # .....

    # latent [n, i, y, x]
    _X_s = [[latent[n, i, y, x] for i=1:Cin, n=1:N] for y=1:H, x=1:W]  # [y, x][i, n]
    # Matrix of matrices(↓)
    #
    # X[i=1 n=1] X[i=1 n=2] ...
    # X[i=2 n=1] X[i=2 n=2]
    # ...                   ...
    #
    # for each (y, x)
    
    # model.enc_wc1
    #               Cout Cin η ξ

    # wc [o, i, η, ξ]
    _W_s = [wc[:, :, η, ξ] for η=1:HH, ξ=1:WW]
    # Matrix of matrices(↓)
    #
    # W[o=1 i=1] W[o=1 i=2] ...
    # W[o=2 i=1] W[o=2 i=2]
    # ...                   ...
    #
    # for each (η, ξ)

    _A_ = [_W_s[η, ξ] * _X_s[y, x] for η = 1:HH, ξ = 1:WW, y = 1:H, x = 1:W]  # [η, ξ, y, x][o, n]
    # 4D tensor of matrices(↓)
    #
    # A[o=1 n=1] A[o=1 n=2] ...
    # A[o=2 n=1] A[o=2 n=2]
    # ...                   ...
    #
    # for each (η, ξ, y, x)

    # _A_ [η, ξ, y, x][o, n]
    # sum_A_ [y, x, n][o]
    # bc [o]

    # [n, o, y, x]

    (
        sum(
            1 ≤ y + Δy ≤ H && 1 ≤ x + Δx ≤ W
            ? _A_[Δy + kw + 1, Δx + kw + 1, y + Δy, x + Δx][o, n] : 0.0f0
            for Δy = -kw:kw, Δx = -kw:kw
        ) + bc[o]
        for n = 1:N, o = 1:Cout, y = 1:H, x = 1:W
    )  # [n, o, y, x]
end

# TODO: Exactly the same as `groupnorm` in main.jl. Unify with it.
function groupnorm(g, t, num_groups, conv)
    N, C, H, W = size(conv)
    @assert C % num_groups == 0
    size_of_group = C ÷ num_groups
    
    conv_mean = [
        mean(
            conv[n, group * size_of_group + igroup, y, x]
            for igroup = 1:size_of_group, y = 1:H, x = 1:W
        )
        for n = 1:N, group = 0:(num_groups - 1)
    ]

    conv_var = [
        var(
            (
                conv[n, group * size_of_group + igroup, y, x]
                for igroup = 1:size_of_group, y = 1:H, x = 1:W
            ), corrected = false
        )
        for n = 1:N, group = 0:(num_groups - 1)
    ]

    [
        (g[o] * (conv[n, o, y, x] - conv_mean[n, (o - 1) ÷ size_of_group + 1]) / √(conv_var[n, (o - 1) ÷ size_of_group + 1] + 1f-5) + t[o])
        for n = 1:N, o = 1:C, y = 1:H, x = 1:W
    ]
end

function calc_drblock(latent, drblock)
    num_n, num_o, num_y, num_x = size(latent)  # latent [n, o, y, x]

    x = groupnorm(drblock.g1, drblock.t1, 32, latent)  # [n, o, y, x]
    x = x ./ (exp.(-x) .+ 1)  # [n, o, y, x]
    x = conv2d(drblock.wc1, drblock.bc1, x) |> collect  # [n, fo, y, x]
    _, num_fo, _, _ = size(x)
    x = groupnorm(drblock.g2, drblock.t2, 32, x)  # [n, fo, y, x]
    x = x ./ (exp.(-x) .+ 1)  # [n, fo, y, x]
    x = conv2d(drblock.wc2, drblock.bc2, x) |> collect  # [n, fo, y, x]
    if num_o == num_fo
        l = latent  # [n, fo, y, x]
    else
        l = conv2d(drblock.wc3, drblock.bc3, latent) |> collect  # [n, fo, y, x]
    end
    l .+ x  # [n, fo, y, x]
end

decref = load_safetensors("../../../Downloads/decref.safetensors")

tensors = load_safetensors("../../../Downloads/sd/v1-5/model.safetensors")
config = JSON.parsefile("../../../Downloads/sd/v1-5/config.json")
model = get_model(tensors, config)

x = decref["l"]
@assert size(x) == (1, 4, 64, 64)
x ./= 0.18215
@assert size(x) == (1, 4, 64, 64)
x = conv2d(model.dconvs["first_stage_model.post_quant_conv"]..., x) |> collect
@assert size(x) == (1, 4, 64, 64)
x = conv2d(model.dconvs["first_stage_model.decoder.conv_in"]..., x) |> collect
@assert size(x) == (1, 512, 64, 64)
x = calc_drblock(x, model.drblocks["first_stage_model.decoder.mid.block_1"]) |> collect
@assert size(x) == (1, 512, 64, 64)
tshow(x)
