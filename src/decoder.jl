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

function layer_norm(x, g, t)
    g .* (x .- mean(x)) ./ √(var(x, corrected = false) + 1f-5) + t
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

# Numerically stable softmax
function softmax(x)
    x = exp.(x .- maximum(x))
    x / sum(x)
end

function calc_dablock(x, ablock)
    num_n, num_o, num_y, num_x = size(x)
    num_xy = num_x * num_y

    latent_ = x  # [n, o, y, x]
    x = groupnorm(ablock.g, ablock.t, 32, x)  # [n, o, y, x]

    x = permutedims(x, (1, 2, 4, 3))  # [n, o, x, y]
    x = reshape(x, (num_n, num_o, num_xy))  # [n, o, xy]
    x = eachslice(x; dims=(1, 3))  #[n, xy][o]
    y = [[x[n, xy][o] for xy=1:num_xy, o=1:num_o] for n=1:num_n]  # [n][xy, o]
    x = [[x[n, xy][o] for o=1:num_o, xy=1:num_xy] for n=1:num_n]  # [n][o, xy]

    n_embd = num_o
    n_head = 8
    size_head = n_embd ÷ n_head
    x1 = [
        begin
            y = ablock.w1 * y + ablock.b1
            qq = y[1:n_embd, :]
            kk = y[(n_embd + 1):(2 * n_embd), :]
            vv = y[(2 * n_embd + 1):(3 * n_embd), :]
            q = (qq[((i - 1) * size_head + 1):(i * size_head), :] for i = 1:n_head)
            k = (kk[((i - 1) * size_head + 1):(i * size_head), :] for i = 1:n_head)
            v = (vv[((i - 1) * size_head + 1):(i * size_head), :] for i = 1:n_head)
            kq = transpose.(k) .* q ./ sqrt(Float32(size_head))
            kqq = [hcat([softmax(kq__) for kq__ in eachcol(kq_)]...) for kq_ in kq]
            v .* kqq
        end
        for y ∈ x
    ]  # [n][head][ihead, xy]

    # x1  # [n][head][ihead, xy]
    a1 = [x1[n][head][ihead, xy] for n=1:num_n, xy=1:num_xy, ihead=1:size_head, head=1:n_head]  # [n, xy, ihead, head]
    a2 = reshape(a1, (num_n, num_xy, num_o))  # [n, xy, o]
    a3 = (a2[n, xy, :] for n=1:num_n, xy=1:num_xy)  # [n, xy][o]
    a4 = Ref(ablock.w2) .* a3 .+ Ref(ablock.b2)  # [n, xy][o]
    x2 = [a4[n, xy][o] for n=1:num_n, xy=1:num_xy, o=1:num_o]  # [n, xy, o]
    d = [x2[n, xy, o] for n=1:num_n, o=1:num_o, xy=1:num_xy]  # [n, o, xy]
    d = reshape(d, (num_n, num_o, num_x, num_y))  # [n, o, x, y]
    latent_ .+ d
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
x = calc_drblock(x, model.drblocks["first_stage_model.decoder.mid.block_1"])
@assert size(x) == (1, 512, 64, 64)
x = calc_dablock(x, model.dablocks["first_stage_model.decoder.mid.attn_1"])
@assert size(x) == (1, 512, 64, 64)
tshow(x)
