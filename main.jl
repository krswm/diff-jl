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

function tshow(x)
    show(IOContext(stdout, :limit => true), "text/plain", x)
    println()
end

# Originally taken from my GPT-2 inference implementation.
function main()::Nothing
    if length(ARGS) ≠ 3
        println("GPT-2 Inference with Julia")
        print("Usage: ")
        printstyled(
            "julia --project $PROGRAM_FILE <path to model repository> <path to model safetensors> <your prompt>",
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

    tensors = load_safetensors(ARGS[2])

    model = get_model(tensors)

    # ==== Tokenization ====

    # Token IDs
    ids = tokenize(token_to_id, ranks, ARGS[3], model.n_ctx)
    ids |> length |> println

    # ==== Inference ====

    id = 0
    k_caches = [
        [Matrix{Float32}(undef, model.n_embd ÷ model.n_head, 0) for _ = 1:model.n_head]
        for _ = 1:model.n_layer
    ]
    v_caches = [
        [Matrix{Float32}(undef, model.n_embd ÷ model.n_head, 0) for _ = 1:model.n_head]
        for _ = 1:model.n_layer
    ]

    prompt_embedding = Matrix{Float32}(undef, model.n_embd, 0)
    for (pos, id) in enumerate(ids)
         prompt_embedding = hcat(
            prompt_embedding,
            transformer!(id, pos, model, k_caches, v_caches),
        )
    end
   prompt_embedding |> tshow
end

main()
