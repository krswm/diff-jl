using Statistics

function layernorm(x, w, b)
    # x [X, Y, C, N]  # w [C]  # b [C]
    e = convert(eltype(x), 1.0e-5)
    w = insertdims(w, dims = (1, 2, 4))                  # [1, 1, C, 1]
    b = insertdims(b, dims = (1, 2, 4))                  # [1, 1, C, 1]
    x_mean = mean(x, dims = (1, 2, 3))                   # [1, 1, 1, N]
    x_var = var(x, corrected = false, dims = (1, 2, 3))  # [1, 1, 1, N]
    w .* (x .- x_mean) ./ .√(x_var .+ e) .+ b            # [X, Y, C, N]
end

function groupnorm(x, w, b, num_G)
    # x [X, Y, C, N]  # w [C]  # b [C]
    num_X, num_Y, num_C, num_N = size(x)
    num_I = num_C ÷ num_G
    e = convert(eltype(x), 1.0e-5)
    w = insertdims(w, dims = (1, 2, 4))                          # [1, 1, C, 1]
    b = insertdims(b, dims = (1, 2, 4))                          # [1, 1, C, 1]
    x_reshape = reshape(x, num_X, num_Y, num_I, num_G, num_N)    # [X, Y, I, G, N]
    x_mean = mean(x_reshape, dims = (1, 2, 3))                   # [1, 1, 1, G, N]
    x_mean = dropdims(x_mean, dims = 3)                          # [1, 1, G, N]
    x_mean = repeat(x_mean, inner = (1, 1, num_I, 1))            # [1, 1, C, N]
    x_var = var(x_reshape, corrected = false, dims = (1, 2, 3))  # [1, 1, 1, G, N]
    x_var = dropdims(x_var, dims = 3)                            # [1, 1, G, N]
    x_var = repeat(x_var, inner = (1, 1, num_I, 1))              # [1, 1, C, N]
    w .* (x .- x_mean) ./ .√(x_var .+ e) .+ b                    # [X, Y, C, N]
end
