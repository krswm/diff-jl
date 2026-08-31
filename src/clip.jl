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

module CLIP

export clip

using JSON
using SafeTensors

using ..Model
using ..Tokenizer
using ..Transformer

function clip()::Matrix{Float32}
    if length(ARGS) ≠ 2
        println("GPT-2 Inference with Julia")
        print("Usage: ")
        printstyled(
            "julia --project $PROGRAM_FILE <path to model repository> <your prompt>",
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
    ids = tokenize(token_to_id, ranks, model, ARGS[2])

    # ==== Inference ====

    k_caches = [
        [Matrix{Float32}(undef, model.n_embd ÷ model.n_head, 0) for _ = 1:model.n_head]
        for _ = 1:model.n_layer
    ]
    v_caches = [
        [Matrix{Float32}(undef, model.n_embd ÷ model.n_head, 0) for _ = 1:model.n_head]
        for _ = 1:model.n_layer
    ]
    x = Matrix{Float32}(undef, model.n_embd, 0)
    for (pos, id) ∈ enumerate(ids)
        x = hcat(x, transformer!(id, pos, model, k_caches, v_caches))
    end
    x
end

end
