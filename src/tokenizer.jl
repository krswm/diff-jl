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

module Tokenizer

export tokenize

using ..Model

# CLIP has a unique encoding.
# e.g.: 'à' (U+00E0) is encoded as "Ãł" (U+00C3 0+0142)
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

# Tokenize `input` with the BPE algorithm.
function tokenize(
    token_to_id::Dict{String,Int},
    ranks::Dict{Tuple{String,String},Int},
    model::Model,
    input::String,
)::Array{Int}
    raw_tokens = string.(split(lowercase(input)))
    raw_tokens = encode_unique_encoding.(raw_tokens)
    ids = Int[]
    push!(ids, token_to_id["<|startoftext|>"])
    for raw_token ∈ raw_tokens
        if haskey(token_to_id, raw_token * "</w>")
            # `raw_token` is already a valid token.
            push!(ids, token_to_id[raw_token*"</w>"])
        else
            # `raw_token` is not a valid token.
            # Split `raw_token` and get valid tokens with the merge algorithm.
            tokens = string.(collect(raw_token))
            tokens[end] *= "</w>"
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
    push!(ids, (token_to_id["<|endoftext|>"] for _ = (length(ids)+1):(model.n_ctx-1))...)
    resize!(ids, model.n_ctx - 1)
    push!(ids, token_to_id["<|endoftext|>"])
    ids
end

end
