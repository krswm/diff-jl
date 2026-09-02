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

module Model

export Layer, Model, get_model

using JSON

struct Layer
    g1::Vector{Float32}
    t1::Vector{Float32}
    w11::Matrix{Float32}
    b11::Vector{Float32}
    w12::Matrix{Float32}
    b12::Vector{Float32}
    g2::Vector{Float32}
    t2::Vector{Float32}
    w21::Matrix{Float32}
    b21::Vector{Float32}
    w22::Matrix{Float32}
    b22::Vector{Float32}
end

struct ResidualBlock
    g1::Vector{Float32}
    t1::Vector{Float32}
    wc1::Array{Float32, 4}
    bc1::Vector{Float32}
    w::Matrix{Float32}
    b::Vector{Float32}
    g2::Vector{Float32}
    t2::Vector{Float32}
    wc2::Array{Float32, 4}
    bc2::Vector{Float32}
    wc3::Array{Float32, 4}
    bc3::Vector{Float32}
end

struct AttentionBlock
    g1::Vector{Float32}
    t1::Vector{Float32}
    wc1::Array{Float32, 4}
    bc1::Vector{Float32}

    g2::Vector{Float32}
    t2::Vector{Float32}
    w21::Matrix{Float32}
    w22::Matrix{Float32}
    b22::Vector{Float32}

    g3::Vector{Float32}
    t3::Vector{Float32}
    w31q::Matrix{Float32}
    w31k::Matrix{Float32}
    w31v::Matrix{Float32}
    w32::Matrix{Float32}
    b32::Vector{Float32}

    g4::Vector{Float32}
    t4::Vector{Float32}
    w41::Matrix{Float32}
    b41::Vector{Float32}
    w42::Matrix{Float32}
    b42::Vector{Float32}
    wc4::Array{Float32, 4}
    bc4::Vector{Float32}
end

struct Model
    n_ctx::Int
    n_embd::Int
    n_head::Int
    n_layer::Int
    vocab_size::Int
    e::Float32
    wte::Matrix{Float32}
    wpe::Matrix{Float32}
    layers::Array{Layer}
    gf::Vector{Float32}
    tf::Vector{Float32}

    time_w1::Matrix{Float32}
    time_b1::Vector{Float32}
    time_w2::Matrix{Float32}
    time_b2::Vector{Float32}

    convs::Dict{String, Tuple{Array{Float32, 4}, Vector{Float32}}}
    rblocks::Dict{String, ResidualBlock}
    ablocks::Dict{String, AttentionBlock}
end

function validate_size(tensor, expected)
    if size(tensor) ≠ expected
        error("size of tensor $(size(tensor)) differs from expected $expected")
    end
end

function get_model(tensors::Dict{String,Array}, config::JSON.Object)::Model
    n_ctx = config["max_position_embeddings"]
    n_embd = config["hidden_size"]
    n_head = config["num_attention_heads"]
    n_layer = config["num_hidden_layers"]
    vocab_size = config["vocab_size"]
    e = Float32(config["layer_norm_eps"])

    prefix = "cond_stage_model.transformer.text_model"
    wte = permutedims(tensors["$prefix.embeddings.token_embedding.weight"])
    validate_size(wte, (n_embd, vocab_size))
    wpe = permutedims(tensors["$prefix.embeddings.position_embedding.weight"])
    validate_size(wpe, (n_embd, n_ctx))
    layers = Layer[]
    for i = 0:(n_layer-1)
        g1 = tensors["$prefix.encoder.layers.$i.layer_norm1.weight"]
        validate_size(g1, (n_embd,))
        t1 = tensors["$prefix.encoder.layers.$i.layer_norm1.bias"]
        validate_size(t1, (n_embd,))
        w11 = vcat(
            tensors["$prefix.encoder.layers.$i.self_attn.q_proj.weight"],
            tensors["$prefix.encoder.layers.$i.self_attn.k_proj.weight"],
            tensors["$prefix.encoder.layers.$i.self_attn.v_proj.weight"],
        )
        validate_size(w11, (n_embd * 3, n_embd))
        b11 = vcat(
            tensors["$prefix.encoder.layers.$i.self_attn.q_proj.bias"],
            tensors["$prefix.encoder.layers.$i.self_attn.k_proj.bias"],
            tensors["$prefix.encoder.layers.$i.self_attn.v_proj.bias"],
        )
        validate_size(b11, (n_embd * 3,))
        w12 = tensors["$prefix.encoder.layers.$i.self_attn.out_proj.weight"]
        validate_size(w12, (n_embd, n_embd))
        b12 = tensors["$prefix.encoder.layers.$i.self_attn.out_proj.bias"]
        validate_size(b12, (n_embd,))
        g2 = tensors["$prefix.encoder.layers.$i.layer_norm2.weight"]
        validate_size(g2, (n_embd,))
        t2 = tensors["$prefix.encoder.layers.$i.layer_norm2.bias"]
        validate_size(t2, (n_embd,))
        w21 = tensors["$prefix.encoder.layers.$i.mlp.fc1.weight"]
        validate_size(w21, (n_embd * 4, n_embd))
        b21 = tensors["$prefix.encoder.layers.$i.mlp.fc1.bias"]
        validate_size(b21, (n_embd * 4,))
        w22 = tensors["$prefix.encoder.layers.$i.mlp.fc2.weight"]
        validate_size(w22, (n_embd, n_embd * 4))
        b22 = tensors["$prefix.encoder.layers.$i.mlp.fc2.bias"]
        validate_size(b22, (n_embd,))
        layer = Layer(g1, t1, w11, b11, w12, b12, g2, t2, w21, b21, w22, b22)
        push!(layers, layer)
    end
    gf = tensors["$prefix.final_layer_norm.weight"]
    validate_size(gf, (n_embd,))
    tf = tensors["$prefix.final_layer_norm.bias"]
    validate_size(tf, (n_embd,))

    prefix = "model.diffusion_model.time_embed"
    time_w1 = tensors["$prefix.0.weight"]
    time_b1 = tensors["$prefix.0.bias"]
    time_w2 = tensors["$prefix.2.weight"]
    time_b2 = tensors["$prefix.2.bias"]

    prefix = "model.diffusion_model.input_blocks"

    convs = Dict(
        key => (tensors["$key.weight"], tensors["$key.bias"]) for key ∈ [
            "model.diffusion_model.input_blocks.0.0",
            "model.diffusion_model.input_blocks.3.0.op",
            "model.diffusion_model.input_blocks.6.0.op",
            "model.diffusion_model.input_blocks.9.0.op",
       ]
    )

    rblocks = Dict(
        key => begin
            g1  = tensors["$key.in_layers.0.weight"]
            t1  = tensors["$key.in_layers.0.bias"]
            wc1 = tensors["$key.in_layers.2.weight"]
            bc1 = tensors["$key.in_layers.2.bias"]
            w   = tensors["$key.emb_layers.1.weight"]
            b   = tensors["$key.emb_layers.1.bias"]
            g2  = tensors["$key.out_layers.0.weight"]
            t2  = tensors["$key.out_layers.0.bias"]
            wc2 = tensors["$key.out_layers.3.weight"]
            bc2 = tensors["$key.out_layers.3.bias"]
            if has_skip_connection
                wc3 = tensors["$key.skip_connection.weight"]
                bc3 = tensors["$key.skip_connection.bias"]
            else
                # Dummies
                wc3 = zeros(0, 0, 0, 0)
                bc3 = zeros(0)
            end
            ResidualBlock(g1, t1, wc1, bc1, w, b, g2, t2, wc2, bc2, wc3, bc3)
        end for (key, has_skip_connection) ∈ [
            ("model.diffusion_model.input_blocks.1.0",  false), 
            ("model.diffusion_model.input_blocks.2.0",  false), 
            ("model.diffusion_model.input_blocks.4.0",  true ), 
            ("model.diffusion_model.input_blocks.5.0",  false), 
            ("model.diffusion_model.input_blocks.7.0",  true ), 
            ("model.diffusion_model.input_blocks.8.0",  false), 
            ("model.diffusion_model.input_blocks.10.0", false), 
            ("model.diffusion_model.input_blocks.11.0", false), 
        ]
    )

    ablocks = Dict(
        key => begin
            g1   = tensors["$key.norm.weight"]
            t1   = tensors["$key.norm.bias"]
            wc1  = tensors["$key.proj_in.weight"]
            bc1  = tensors["$key.proj_in.bias"]
    
            g2   = tensors["$key.transformer_blocks.0.norm1.weight"]
            t2   = tensors["$key.transformer_blocks.0.norm1.bias"]
            w21  = vcat(
                tensors["$key.transformer_blocks.0.attn1.to_q.weight"],
                tensors["$key.transformer_blocks.0.attn1.to_k.weight"],
                tensors["$key.transformer_blocks.0.attn1.to_v.weight"],
            )
            w22  = tensors["$key.transformer_blocks.0.attn1.to_out.0.weight"]
            b22  = tensors["$key.transformer_blocks.0.attn1.to_out.0.bias"]

            g3   = tensors["$key.transformer_blocks.0.norm2.weight"]
            t3   = tensors["$key.transformer_blocks.0.norm2.bias"]
            w31q = tensors["$key.transformer_blocks.0.attn2.to_q.weight"]
            w31k = tensors["$key.transformer_blocks.0.attn2.to_k.weight"]
            w31v = tensors["$key.transformer_blocks.0.attn2.to_v.weight"]
            w32  = tensors["$key.transformer_blocks.0.attn2.to_out.0.weight"] 
            b32  = tensors["$key.transformer_blocks.0.attn2.to_out.0.bias"] 

            g4   = tensors["$key.transformer_blocks.0.norm3.weight"]
            t4   = tensors["$key.transformer_blocks.0.norm3.bias"]
            w41  = tensors["$key.transformer_blocks.0.ff.net.0.proj.weight"]
            b41  = tensors["$key.transformer_blocks.0.ff.net.0.proj.bias"]
            w42  = tensors["$key.transformer_blocks.0.ff.net.2.weight"]
            b42  = tensors["$key.transformer_blocks.0.ff.net.2.bias"]
            wc4  = tensors["$key.proj_out.weight"]
            bc4  = tensors["$key.proj_out.bias"]

            AttentionBlock(
                g1, t1, wc1, bc1,
                g2, t2, w21, w22, b22,
                g3, t3, w31q, w31k, w31v, w32, b32,
                g4, t4, w41, b41, w42, b42, wc4, bc4,
            )
        end for key ∈ [
            "model.diffusion_model.input_blocks.1.1",
            "model.diffusion_model.input_blocks.2.1",
            "model.diffusion_model.input_blocks.4.1",
            "model.diffusion_model.input_blocks.5.1",
            "model.diffusion_model.input_blocks.7.1",
            "model.diffusion_model.input_blocks.8.1",
        ]
    )

    Model(
        n_ctx,
        n_embd,
        n_head,
        n_layer,
        vocab_size,
        e,
        wte,
        wpe,
        layers,
        gf,
        tf,
        time_w1,
        time_b1,
        time_w2,
        time_b2,
        convs,
        rblocks,
        ablocks,
    )
end

end
