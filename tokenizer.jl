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

# Originally taken from my GPT-2 tokenizer implementation.
module Tokenizer

export decode_unique_encoding!, tokenize

# GPT-2 has a unique encoding.
# e.g.: 'Ġ' (U+0120) → 0x20

function encode_unique_encoding(text::String)::String
    encoded = [
        if byte ∈ 0x00:0x20
            UInt32(byte + 0x0100)
        elseif byte ∈ 0x21:0x7E
            UInt32(byte)
        elseif byte ∈ 0x7F:0xA0
            UInt32(byte + 0x00A2)
        elseif byte ∈ 0xA1:0xAC
            UInt32(byte)
        elseif byte == 0xAD
            0x00000143
        elseif byte ∈ 0xAE:0xFF
            UInt32(byte)
        end for byte ∈ transcode(UInt8, text)
    ]
    transcode(String, encoded)
end

function decode_unique_encoding!(encoded::String, buffer::Array{UInt8})::String
    bytes = [
        if codepoint ∈ 0x0100:0x0120
            UInt8(codepoint - 0x0100)
        elseif codepoint ∈ 0x0021:0x007E
            UInt8(codepoint)
        elseif codepoint ∈ 0x0121:0x0142
            UInt8(codepoint - 0x00A2)
        elseif codepoint ∈ 0x00A1:0x00AC
            UInt8(codepoint)
        elseif codepoint == 0x0143
            0xAD
        elseif codepoint ∈ 0x00AE:0x00FF
            UInt8(codepoint)
        end for codepoint ∈ transcode(UInt32, encoded)
    ]
    bytes = vcat(buffer, bytes)
    empty!(buffer)

    # A token may contain only a part of UTF-8 sequence.
    # Decode it incrementally.

    # Unfinished valid UTF-8 sequences:
    #
    # Case A1: 110xxxxx
    #
    # Case B1: 1110xxxx
    # Case B2: 1110xxxx 10xxxxxx
    #
    # Case C1: 11110xxx
    # Case C2: 11110xxx 10xxxxxx
    # Case C3: 11110xxx 10xxxxxx 10xxxxxx

    if length(bytes) ≥ 3 &&
       bytes[end-2] ∈ 0xC0:0xDF &&
       bytes[end-1] ∈ 0x80:0xBF &&
       bytes[end] ∈ 0x80:0xBF
        # Case C3
        decoded = bytes[1:(end-3)]
        append!(buffer, bytes[(end-2):end])
    elseif length(bytes) ≥ 2 && bytes[end-1] ∈ 0xC0:0xEF && bytes[end] ∈ 0x80:0xBF
        # Case B2 and Case C2
        decoded = bytes[1:(end-2)]
        append!(buffer, bytes[(end-1):end])
    elseif length(bytes) ≥ 1 && bytes[end] ∈ 0xC0:0xF7
        # Case A1, Case B1, and Case C1
        decoded = bytes[1:(end-1)]
        append!(buffer, bytes[end:end])
    else
        # No unfinished sequence at the end
        decoded = bytes
    end

    decoded = transcode(String, decoded)
    join((valid ? char : '�') for (char, valid) ∈ zip(decoded, isvalid.(collect(decoded))))
end

# Tokenize `input` with the BPE algorithm.
function tokenize(
    token_to_id::Dict{String,Int},
    ranks::Dict{Tuple{String,String},Int},
    input::String,
    fill_size::Int
)::Array{Int}
    raw_tokens = split(input)
    raw_tokens = raw_tokens .* "</w>"  # End of word
    raw_tokens = encode_unique_encoding.(raw_tokens)

    startoftext_id = token_to_id["<|startoftext|>"]
    endoftext_id = token_to_id["<|endoftext|>"]
    ids = Int[startoftext_id]  # Token IDs
    for raw_token ∈ raw_tokens
        if haskey(token_to_id, raw_token)
            # `raw_token` is already a valid token.
            push!(ids, token_to_id[raw_token])
        else
            # `raw_token` is not a valid token.
            # Split `raw_token` and get valid tokens with the merge algorithm.
            tokens = string.(collect(raw_token))
            while length(tokens) ≥ 2
                pairs = zip(tokens[1:(end-1)], tokens[2:end])
                best_rank = typemax(Int)
                best_i_pair = typemax(Int)
                for (i_pair, pair) ∈ enumerate(pairs)
                    if haskey(ranks, pair) && ranks[pair] < best_rank
                        best_rank = ranks[pair]
                        best_i_pair = i_pair
                    end
                end
                if best_i_pair == typemax(Int)
                    break
                end

                tokens[best_i_pair] *= tokens[best_i_pair+1]
                deleteat!(tokens, best_i_pair + 1)
            end

            for token ∈ tokens
                push!(ids, token_to_id[token])
            end
        end
    end
    for _ = length(ids):fill_size
        push!(ids, endoftext_id)
    end
    ids
end

end

using JSON
using SafeTensors

include("model.jl")
using .Model
using .Tokenizer
include("transformer.jl")
using .Transformer

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
    ids = tokenize(token_to_id, ranks, ARGS[3], 77)
    if length(ids) == 0
        println("Your prompt should not be empty.")
        exit()
    end

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

    # This time the transformer will used not for text generation.
    # I'll use it just for the prompt embedding.
    # (Can I use it for text generation, technically?)
    for (pos, id) in enumerate(ids)
        logits = transformer!(id, pos, model, k_caches, v_caches)
    end
end

main()

# Apparently it doesn't work without changing anything.
# Things that's different from GPT-2 I noticed:
# - The `<|startoftext|>` and the `<|endoftext|>` tokens
# - `</w>` stuff (Maybe end of word? It resembles HTML and XML end markers.)

# What if the prompt includes "\n"?
# -> If I test Hugging Face's `tokenizer.batch_encode_plus` (Python)
#    I observed it removes such bytes.
#    e.g. -> "A\nB" and "A B" are the same
# Julia's `split` will work.

# Completed!
