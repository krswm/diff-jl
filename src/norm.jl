function layernorm(x, w, b)
    # x [x, y, c, n]
    # w [c]
    # t [c]
    w = insertdims(w, dims=(1, 2, 4))  # [1, 1, c, 1]
    b = insertdims(b, dims=(1, 2, 4))  # [1, 1, c, 1]
    e = convert(eltype(x), 1.0e-5)
    x_mean = mean(x, dims=(1, 2, 3))  # [1, 1, 1, n]
    x_var = var(x, corrected=false, dims=(1, 2, 3))  # [1, 1, 1, n]
    w .* (x .- x_mean) ./ .√(x_var .+ e) .+ b  # [x, y, c, n]
end
