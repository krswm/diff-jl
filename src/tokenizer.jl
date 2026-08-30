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

# U+00C0 (C3 80) -> U+00E0 U+00AF

# Hmm, CLIP also has "unique encoding"
# but it's not identical to GPT-2's.
# Let me reverse-engineer Hugging Face's tokenizer! 
# (I won't see the source code of HF's tokenizer. It's my challege for myself!)
# ...
# Nevermind, they are actually identical. It is my mistake :(
# There's a bug on MY code.
# ...
# Well, actually they ARE different.
# It uses U+00A1-U+0143, which is the same range GPT-2 uses
# but CLIP seems to have a different mapping.
# ...
# It turns out U+00C0 is coincidentally somewhat a rara case
# because it supposed to use U+00A0 for the second UTF-8 byte (0x80)
# but U+00A0 is nbsp and it's avoided (maybe because it's invisible, I guess)
# so it uses U+0142 instead.
# ...
# Wait wait wait, Capital letters are converted to small letters!?
# I haven't realized that because I've used "cat with hat"
# as the debug test prompt!
# This explains everything.
# U+00C0 is À and its lower case counterpart is à.
# "à" encoded with my GPT-2 tokenizer implementation is indeed "\ue0\uef".
# à la carte?
# The conclusion is that CLIP's unique encoding is same as GPT-2 under the hood
# the only difference is for CLiP the text is turned lowercase beforehand.

# Tokenize `input` with the BPE algorithm.
function tokenize(
    token_to_id::Dict{String,Int},
    ranks::Dict{Tuple{String,String},Int},
    input::String,
    fill_size::Int
)::Array{Int}
    input = lowercase(input)
    raw_tokens = String.(split(input))
    raw_tokens = encode_unique_encoding.(raw_tokens)

    startoftext_id = token_to_id["<|startoftext|>"]
    endoftext_id = token_to_id["<|endoftext|>"]
    ids = Int[startoftext_id]  # Token IDs
    for raw_token ∈ raw_tokens
        if haskey(token_to_id, raw_token * "</w>")
            # `raw_token` is already a valid token.
            push!(ids, token_to_id[raw_token * "</w>"])
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
    for _ = (length(ids) + 1):fill_size
        push!(ids, endoftext_id)
    end
    ids
    ids |> println
    exit(0)
end

end
