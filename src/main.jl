# GPT-2 inference with Julia
# Copyright (C) 2026  Kurosawa Mutsumi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

using JSON
using SafeTensors

include("model.jl")
using .Model
include("tokenizer.jl")
using .Tokenizer
include("transformer.jl")
using .Transformer

function get_prompt_embedding(ids::Vector{Int}, model::Model.Model)::Matrix{Float32}
    k_caches = [
        [Matrix{Float32}(undef, model.n_embd ÷ model.n_head, 0) for _ = 1:model.n_head]
        for _ = 1:model.n_layer
    ]
    v_caches = [
        [Matrix{Float32}(undef, model.n_embd ÷ model.n_head, 0) for _ = 1:model.n_head]
        for _ = 1:model.n_layer
    ]
    x = Vector{Float32}[]
    for (pos, id) ∈ enumerate(ids)
        push!(x, transformer!(id, pos, model, k_caches, v_caches))
    end
    x = hcat(x...)
    x
end

function main()::Nothing
    if length(ARGS) ≠ 4
        println("Machine learning stuff with Julia")
        print("Usage: ")
        printstyled(
            "julia --project $PROGRAM_FILE <path to model repository> <path to pre-sampled random tensors> <your positive prompt> <your negative prompt>",
            bold = true,
        )
        println()
        println("You may have to enclose 'your prompt' with quotes.")
        exit()
    end

    token_to_id, id_to_token = begin
        vocab = JSON.parsefile("$(ARGS[1])/vocab.json")
        token_to_id = Dict(token => id for (token, id) ∈ vocab)
        id_to_token = Dict(id => token for (token, id) ∈ vocab)
        token_to_id, id_to_token
    end

    ranks = begin
        ranks = Dict{Tuple{String,String},Int}()
        rank = 0
        for line ∈ readlines("$(ARGS[1])/merges.txt")
            # Skip a comment line.
            if startswith(line, "#")
                continue
            end

            token0, token1 = split(line, " ")
            ranks[(token0, token1)] = rank
            rank += 1
        end
        ranks
    end

    model = begin
        tensors = load_safetensors("$(ARGS[1])/model.safetensors")
        config = JSON.parsefile("$(ARGS[1])/config.json")
        get_model(tensors, config)
    end

    # ==== Tokenization ====

    # Token IDs
    positive_ids = tokenize(token_to_id, ranks, model, ARGS[3])
    negative_ids = tokenize(token_to_id, ranks, model, ARGS[4])

    # ==== Inference ====

    positive_prompt_embedding = get_prompt_embedding(positive_ids, model)
    negative_prompt_embedding = get_prompt_embedding(negative_ids, model)
    c = cat(positive_prompt_embedding, negative_prompt_embedding, dims = 3)

    # ====

    # Pre-sampled random tensors
    # The diffusion model requires a random noise,
    # however, I want the whole tensor calculation deterministic
    # so that I can compare the result with the reference implementation.
    # I extracted the random tensors from reference implementation 
    # (https://github.com/hkproj/pytorch-stable-diffusion, excellent explanation video!)
    # and save it to a Safetensors file.
    # I'll use genuine random number generator in Julia later.
    rand42 = load_safetensors(ARGS[2])

    for t = 900:-100:0
        x = permutedims(rand42["l"], (4, 3, 2, 1))

        x = cat(x, x, dims = 4)

        f = t .* 10000 .^ (0.0f0:(-1.0f0/160):(-159.0f0/160))
        f = vcat(cos.(f), sin.(f))

        f = model.time_w1 * f + model.time_b1
        f = f ./ (exp.(-f) .+ 1)
        f = model.time_w2 * f + model.time_b2

        x |> size |> println
        model.enc_wc1 |> size |> println
        model.enc_bc1 |> size |> println

        println("z start")
        z = [
            model.enc_wc1i[ξ, η, k, Cout] .* x[:, :, k, N]
            for ξ = 1:3, η = 1:3, k = 1:4, Cout = 1:130, N = 1:2
        ]
        println("z end")

        println("start")
        y = [
            sum(
                1 ≤ X + ΔX ≤ 64 && 1 ≤ Y + ΔY ≤ 64
                ? z[ΔX + 2, ΔY + 2, k, Cout, N]
                : 0.0f0
                for ΔX = -1:1, ΔY = -1:1, k = 1:4
            ) + model.enc_bc1[Cout]
            for X = 1:64, Y = 1:64, Cout = 1:320, N = 1:2
        ]

        println("end")
        # At least it gets correct result but it's super slow :(

        show(IOContext(stdout, :limit => true), "text/plain", y)
        println()

        exit()
    end
end

main()
