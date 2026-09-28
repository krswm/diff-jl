# Stable Diffusion Inference with Julia
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

# %%
using JSON
using SafeTensors
using Statistics

# %%
include("model.jl")
using .Model
include("tokenizer.jl")
using .Tokenizer
include("transformer.jl")
using .Transformer
include("ddpm.jl")
using .DDPM

# %%
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

# %%
if length(ARGS) ≠ 4
    println("Stable Diffusion Inference with Julia")
    print("Usage: ")
    printstyled(
        "julia --project $PROGRAM_FILE <path to model repository> <output path> <positive prompt> <negative prompt>",
        bold = true,
    )
    println()
    println("The AI-generated image will be saved to <output path>.")
    println("The image will be what <positive prompt> describes.")
    println("The image will not be what <negative prompt> describes.")
    println("You can leave <negative prompt> empty: ''")
    println("You may have to enclose 'the prompts' with quotes.")
    exit()
end

start_time = time_ns()

println("==== CLIP: Obtaining Context Tensor from Your Prompt... ====")

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
        global rank  # Temporary workaronud. I'll remove it when I wrap the code with `main` again.

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
;

include("decoder_.jl")

generate_image(c, ARGS[1], ARGS[2])

println("time taken: $((time_ns() - start_time) * 1e-9) s")

