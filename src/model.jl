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

    enc_wc1::Array{Float32, 4}
    enc_bc1::Vector{Float32}

    enc_gg1::Vector{Float32}
    enc_tg1::Vector{Float32}

    enc_wc2::Array{Float32, 4}
    enc_bc2::Vector{Float32}

    enc_time_w1::Matrix{Float32}
    enc_time_b1::Vector{Float32}

    g_1_0_out_layers_0::Vector{Float32}
    t_1_0_out_layers_0::Vector{Float32}
    wc_1_0_out_layers_3::Array{Float32, 4}
    bc_1_0_out_layers_3::Vector{Float32}

    g_1_1_norm::Vector{Float32}
    t_1_1_norm::Vector{Float32}
    wc_1_1_proj_in::Array{Float32, 4}
    bc_1_1_proj_in::Vector{Float32}

    g_1_1_transformer_blocks_0_norm1::Vector{Float32}
    t_1_1_transformer_blocks_0_norm1::Vector{Float32}

    w_1_1_transformer_blocks_0_attn1::Matrix{Float32}
    w_1_1_transformer_blocks_0_attn1_to_out_0::Matrix{Float32}
    b_1_1_transformer_blocks_0_attn1_to_out_0::Vector{Float32}

    g_1_1_transformer_blocks_0_norm2::Vector{Float32}
    t_1_1_transformer_blocks_0_norm2::Vector{Float32}

    w_1_1_transformer_blocks_0_attn2_to_q::Matrix{Float32}
    w_1_1_transformer_blocks_0_attn2_to_k::Matrix{Float32}
    w_1_1_transformer_blocks_0_attn2_to_v::Matrix{Float32}
    w_1_1_transformer_blocks_0_attn2_to_out_0::Matrix{Float32}
    b_1_1_transformer_blocks_0_attn2_to_out_0::Vector{Float32}

    g_1_1_transformer_blocks_0_norm3::Vector{Float32}
    t_1_1_transformer_blocks_0_norm3::Vector{Float32}

    # Let me switch back to use short names!
    w_geglu1::Matrix{Float32}
    b_geglu1::Vector{Float32}
    w_geglu2::Matrix{Float32}
    b_geglu2::Vector{Float32}
    wc_conv_out::Array{Float32, 4}
    bc_conv_out::Vector{Float32}
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
    enc_wc1 = tensors["$prefix.0.0.weight"]
    enc_bc1 = tensors["$prefix.0.0.bias"]

    enc_gg1 = tensors["$prefix.1.0.in_layers.0.weight"]
    enc_tg1 = tensors["$prefix.1.0.in_layers.0.bias"]

    enc_wc2 = tensors["$prefix.1.0.in_layers.2.weight"]
    enc_bc2 = tensors["$prefix.1.0.in_layers.2.bias"]

    enc_time_w1 = tensors["$prefix.1.0.emb_layers.1.weight"]
    enc_time_b1 = tensors["$prefix.1.0.emb_layers.1.bias"]

    g_1_0_out_layers_0 = tensors["$prefix.1.0.out_layers.0.weight"]
    t_1_0_out_layers_0 = tensors["$prefix.1.0.out_layers.0.bias"]
    wc_1_0_out_layers_3 = tensors["$prefix.1.0.out_layers.3.weight"]
    bc_1_0_out_layers_3 = tensors["$prefix.1.0.out_layers.3.bias"]


    g_1_1_norm = tensors["$prefix.1.1.norm.weight"]
    t_1_1_norm = tensors["$prefix.1.1.norm.bias"]
    wc_1_1_proj_in = tensors["$prefix.1.1.proj_in.weight"]
    bc_1_1_proj_in = tensors["$prefix.1.1.proj_in.bias"]
    
    g_1_1_transformer_blocks_0_norm1 = tensors[
        "$prefix.1.1.transformer_blocks.0.norm1.weight"
    ]
    t_1_1_transformer_blocks_0_norm1 = tensors[
        "$prefix.1.1.transformer_blocks.0.norm1.bias"
    ]

    w_1_1_transformer_blocks_0_attn1 = vcat(
        tensors["$prefix.1.1.transformer_blocks.0.attn1.to_q.weight"],
        tensors["$prefix.1.1.transformer_blocks.0.attn1.to_k.weight"],
        tensors["$prefix.1.1.transformer_blocks.0.attn1.to_v.weight"],
    )
    w_1_1_transformer_blocks_0_attn1_to_out_0 = (
        tensors["$prefix.1.1.transformer_blocks.0.attn1.to_out.0.weight"]
    )
    b_1_1_transformer_blocks_0_attn1_to_out_0 = (
        tensors["$prefix.1.1.transformer_blocks.0.attn1.to_out.0.bias"]
    )

    g_1_1_transformer_blocks_0_norm2 = tensors[
        "$prefix.1.1.transformer_blocks.0.norm2.weight"
    ]
    t_1_1_transformer_blocks_0_norm2 = tensors[
        "$prefix.1.1.transformer_blocks.0.norm2.bias"
    ]

    
    w_1_1_transformer_blocks_0_attn2_to_q     = tensors["$prefix.1.1.transformer_blocks.0.attn2.to_q.weight"]
    w_1_1_transformer_blocks_0_attn2_to_k     = tensors["$prefix.1.1.transformer_blocks.0.attn2.to_k.weight"]
    w_1_1_transformer_blocks_0_attn2_to_v     = tensors["$prefix.1.1.transformer_blocks.0.attn2.to_v.weight"]
    w_1_1_transformer_blocks_0_attn2_to_out_0 = tensors["$prefix.1.1.transformer_blocks.0.attn2.to_out.0.weight"] 
    b_1_1_transformer_blocks_0_attn2_to_out_0 = tensors["$prefix.1.1.transformer_blocks.0.attn2.to_out.0.bias"] 

    g_1_1_transformer_blocks_0_norm3 = tensors[
        "$prefix.1.1.transformer_blocks.0.norm3.weight"
    ]
    t_1_1_transformer_blocks_0_norm3 = tensors[
        "$prefix.1.1.transformer_blocks.0.norm3.bias"
    ]

    w_geglu1    = tensors["$prefix.1.1.transformer_blocks.0.ff.net.0.proj.weight"]
    b_geglu1    = tensors["$prefix.1.1.transformer_blocks.0.ff.net.0.proj.bias"]
    w_geglu2    = tensors["$prefix.1.1.transformer_blocks.0.ff.net.2.weight"]
    b_geglu2    = tensors["$prefix.1.1.transformer_blocks.0.ff.net.2.bias"]
    wc_convout = tensors["$prefix.1.1.proj_out.weight"]
    bc_convout = tensors["$prefix.1.1.proj_out.bias"]

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
        enc_wc1,
        enc_bc1,
        enc_gg1,
        enc_tg1,
        enc_wc2,
        enc_bc2,
        enc_time_w1,
        enc_time_b1,
        g_1_0_out_layers_0,
        t_1_0_out_layers_0,
        wc_1_0_out_layers_3,
        bc_1_0_out_layers_3,
        g_1_1_norm,
        t_1_1_norm,
        wc_1_1_proj_in,
        bc_1_1_proj_in,
        g_1_1_transformer_blocks_0_norm1,
        t_1_1_transformer_blocks_0_norm1,
        w_1_1_transformer_blocks_0_attn1,
        w_1_1_transformer_blocks_0_attn1_to_out_0,
        b_1_1_transformer_blocks_0_attn1_to_out_0,
        g_1_1_transformer_blocks_0_norm2,
        t_1_1_transformer_blocks_0_norm2,
        w_1_1_transformer_blocks_0_attn2_to_q,
        w_1_1_transformer_blocks_0_attn2_to_k,
        w_1_1_transformer_blocks_0_attn2_to_v,
        w_1_1_transformer_blocks_0_attn2_to_out_0,
        b_1_1_transformer_blocks_0_attn2_to_out_0,
        g_1_1_transformer_blocks_0_norm3,
        t_1_1_transformer_blocks_0_norm3,
        w_geglu1,
        b_geglu1,
        w_geglu2,
        b_geglu2,
        wc_convout,
        bc_convout,
    )
end

end
