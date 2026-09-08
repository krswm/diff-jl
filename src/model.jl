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

export Layer, Model, get_dmodel, get_fmodel, get_model

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

struct DecoderResidualBlock
    g1::Vector{Float32}
    t1::Vector{Float32}
    wc1::Array{Float32, 4}
    bc1::Vector{Float32}
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

struct DecoderAttentionBlock
    g::Vector{Float32}
    t::Vector{Float32}
    w1::Matrix{Float32}
    b1::Matrix{Float32}
    w2::Matrix{Float32}
    b2::Matrix{Float32}
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
    g_final::Vector{Float32}
    t_final::Vector{Float32}
    wc_final::Array{Float32, 4}
    bc_final::Vector{Float32}

    dconvs::Dict{String, Tuple{Array{Float32, 4}, Vector{Float32}}}
    drblocks::Dict{String, DecoderResidualBlock}
    dablocks::Dict{String, DecoderAttentionBlock}
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
            "model.diffusion_model.output_blocks.2.1.conv",
            "model.diffusion_model.output_blocks.5.2.conv",
            "model.diffusion_model.output_blocks.8.2.conv",
       ]
    )

    dconvs = Dict(
        key => (tensors["$key.weight"], tensors["$key.bias"]) for key ∈ [
            "first_stage_model.post_quant_conv",
            "first_stage_model.decoder.conv_in",
            "first_stage_model.decoder.up.3.upsample.conv",
            "first_stage_model.decoder.up.2.upsample.conv",
            "first_stage_model.decoder.up.1.upsample.conv",
            "first_stage_model.decoder.conv_out",
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
            ("model.diffusion_model.input_blocks.1.0",   false),
            ("model.diffusion_model.input_blocks.2.0",   false),
            ("model.diffusion_model.input_blocks.4.0",   true ),
            ("model.diffusion_model.input_blocks.5.0",   false),
            ("model.diffusion_model.input_blocks.7.0",   true ),
            ("model.diffusion_model.input_blocks.8.0",   false),
            ("model.diffusion_model.input_blocks.10.0",  false),
            ("model.diffusion_model.input_blocks.11.0",  false),
            ("model.diffusion_model.middle_block.0",     false),
            ("model.diffusion_model.middle_block.2",     false),
            ("model.diffusion_model.output_blocks.0.0",  true ),
            ("model.diffusion_model.output_blocks.1.0",  true ),
            ("model.diffusion_model.output_blocks.2.0",  true ),
            ("model.diffusion_model.output_blocks.3.0",  true ),
            ("model.diffusion_model.output_blocks.4.0",  true ),
            ("model.diffusion_model.output_blocks.5.0",  true ),
            ("model.diffusion_model.output_blocks.6.0",  true ),
            ("model.diffusion_model.output_blocks.7.0",  true ),
            ("model.diffusion_model.output_blocks.8.0",  true ),
            ("model.diffusion_model.output_blocks.9.0",  true ),
            ("model.diffusion_model.output_blocks.10.0", true ),
            ("model.diffusion_model.output_blocks.11.0", true ),
        ]
    )

    drblocks = Dict(
        key => begin
            g1  = tensors["$key.norm1.weight"]
            t1  = tensors["$key.norm1.bias"]
            wc1 = tensors["$key.conv1.weight"]
            bc1 = tensors["$key.conv1.bias"]
            g2  = tensors["$key.norm2.weight"]
            t2  = tensors["$key.norm2.bias"]
            wc2 = tensors["$key.conv2.weight"]
            bc2 = tensors["$key.conv2.bias"]
            if has_skip_connection
                wc3 = tensors["$key.nin_shortcut.weight"]
                bc3 = tensors["$key.nin_shortcut.bias"]
            else
                # Dummies
                wc3 = zeros(0, 0, 0, 0)
                bc3 = zeros(0)
            end
            DecoderResidualBlock(g1, t1, wc1, bc1, g2, t2, wc2, bc2, wc3, bc3)
        end for (key, has_skip_connection) ∈ [
            ("first_stage_model.decoder.mid.block_1", false),
            ("first_stage_model.decoder.mid.block_2", false),
            ("first_stage_model.decoder.up.3.block.0", false),
            ("first_stage_model.decoder.up.3.block.1", false),
            ("first_stage_model.decoder.up.3.block.2", false),
            ("first_stage_model.decoder.up.2.block.0", false),
            ("first_stage_model.decoder.up.2.block.1", false),
            ("first_stage_model.decoder.up.2.block.2", false),
            ("first_stage_model.decoder.up.1.block.0", true),
            ("first_stage_model.decoder.up.1.block.1", false),
            ("first_stage_model.decoder.up.1.block.2", false),
            ("first_stage_model.decoder.up.0.block.0", true),
            ("first_stage_model.decoder.up.0.block.1", false),
            ("first_stage_model.decoder.up.0.block.2", false),
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
            "model.diffusion_model.middle_block.1",
            "model.diffusion_model.output_blocks.3.1",
            "model.diffusion_model.output_blocks.4.1",
            "model.diffusion_model.output_blocks.5.1",
            "model.diffusion_model.output_blocks.6.1",
            "model.diffusion_model.output_blocks.7.1",
            "model.diffusion_model.output_blocks.8.1",
            "model.diffusion_model.output_blocks.9.1",
            "model.diffusion_model.output_blocks.10.1",
            "model.diffusion_model.output_blocks.11.1",
        ]
    )

    dablocks = Dict(
        key => begin
            g = tensors["$key.norm.weight"]
            t = tensors["$key.norm.bias"]
            w1  = cat(
                tensors["$key.q.weight"],
                tensors["$key.k.weight"],
                tensors["$key.v.weight"];
                dims=1,
            )
            @assert size(w1) == (1536, 512, 1, 1)
            w1 = reshape(w1, (1536, 512))
            w1 = permutedims(w1, (2, 1))
            b1  = cat(
                tensors["$key.q.bias"],
                tensors["$key.k.bias"],
                tensors["$key.v.bias"],
                dims=1,
            )
            b1 = insertdims(b1, dims=1)
            w2 = tensors["$key.proj_out.weight"]
            @assert size(w2) == (512, 512, 1, 1)
            w2 = reshape(w2, (512, 512))
            w2 = permutedims(w2, (2, 1))
            b2 = tensors["$key.proj_out.bias"]
            b2 = insertdims(b2, dims=1)
            DecoderAttentionBlock(g, t, w1, b1, w2, b2)
        end for key ∈ [
            "first_stage_model.decoder.mid.attn_1"
        ]
    )

    g_final  = tensors["model.diffusion_model.out.0.weight"]
    t_final  = tensors["model.diffusion_model.out.0.bias"]
    wc_final = tensors["model.diffusion_model.out.2.weight"]
    bc_final = tensors["model.diffusion_model.out.2.bias"]

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
        g_final,
        t_final,
        wc_final,
        bc_final,
        dconvs,
        drblocks,
        dablocks,
    )
end

# Decoder convolution
struct Dconv
    wc::Array{Float32, 4}  # [x, y, i, o]
    bc::Array{Float32, 4}  # [x, y, o, n]
end

function get_dconv(tensors, prefix)
    wc = tensors["$prefix.weight"]  # [o, i, y, x]
    wc = permutedims(wc, (4, 3, 2, 1))  # [x, y, i, o]
    bc = tensors["$prefix.bias"]  # [o]
    bc = insertdims(bc, dims=(1, 2, 4))  # [x, y, o, n]
    Dconv(wc, bc)
end

# Decoder residual block
struct Drblock
    g1::Vector{Float32}
    t1::Vector{Float32}
    wc1::Array{Float32, 4}
    bc1::Array{Float32, 4}
    g2::Vector{Float32}
    t2::Vector{Float32}
    wc2::Array{Float32, 4}
    bc2::Array{Float32, 4}
end

function get_drblock(tensors, prefix)
    g1  = tensors["$prefix.norm1.weight"]
    t1  = tensors["$prefix.norm1.bias"]
    wc1 = tensors["$prefix.conv1.weight"]  # [o, i, y, x]
    wc1 = permutedims(wc1, (4, 3, 2, 1))  # [x, y, i, o]
    bc1 = tensors["$prefix.conv1.bias"]
    bc1 = insertdims(bc1, dims=(1, 2, 4))  # [x, y, o, n]
    g2  = tensors["$prefix.norm2.weight"]
    t2  = tensors["$prefix.norm2.bias"]
    wc2 = tensors["$prefix.conv2.weight"]  # [o, i, y, x]
    wc2 = permutedims(wc2, (4, 3, 2, 1))  # [x, y, i, o]
    bc2 = tensors["$prefix.conv2.bias"]  # [x, y, o, n]
    bc2 = insertdims(bc2, dims=(1, 2, 4))  # [x, y, o, n]
    Drblock(g1, t1, wc1, bc1, g2, t2, wc2, bc2)
end

# Decoder residual block w/ additional convolution
struct Drcblock
    g1::Vector{Float32}
    t1::Vector{Float32}
    wc1::Array{Float32, 4}
    bc1::Array{Float32, 4}
    g2::Vector{Float32}
    t2::Vector{Float32}
    wc2::Array{Float32, 4}
    bc2::Array{Float32, 4}
    wc3::Array{Float32, 4}
    bc3::Array{Float32, 4}
end

function get_drcblock(tensors, prefix)
    g1  = tensors["$prefix.norm1.weight"]
    t1  = tensors["$prefix.norm1.bias"]
    wc1 = tensors["$prefix.conv1.weight"]  # [o, i, y, x]
    wc1 = permutedims(wc1, (4, 3, 2, 1))  # [x, y, i, o]
    bc1 = tensors["$prefix.conv1.bias"]
    bc1 = insertdims(bc1, dims=(1, 2, 4))  # [x, y, o, n]
    g2  = tensors["$prefix.norm2.weight"]
    t2  = tensors["$prefix.norm2.bias"]
    wc2 = tensors["$prefix.conv2.weight"]  # [o, i, y, x]
    wc2 = permutedims(wc2, (4, 3, 2, 1))  # [x, y, i, o]
    bc2 = tensors["$prefix.conv2.bias"]  # [x, y, o, n]
    bc2 = insertdims(bc2, dims=(1, 2, 4))  # [x, y, o, n]
    wc3 = tensors["$prefix.nin_shortcut.weight"]  # [o, i, y, x]
    wc3 = permutedims(wc3, (4, 3, 2, 1))  # [x, y, i, o]
    bc3 = tensors["$prefix.nin_shortcut.bias"]  # [x, y, o, n]
    bc3 = insertdims(bc3, dims=(1, 2, 4))  # [x, y, o, n]
    Drcblock(g1, t1, wc1, bc1, g2, t2, wc2, bc2, wc3, bc3)
end

struct Dablock
    g::Vector{Float32}
    t::Vector{Float32}
    w1::Matrix{Float32}
    b1::Matrix{Float32}
    w2::Matrix{Float32}
    b2::Matrix{Float32}
end

function get_dablock(tensors, prefix)
        g = tensors["$prefix.norm.weight"]
        t = tensors["$prefix.norm.bias"]
        w1  = cat(
            tensors["$prefix.q.weight"],
            tensors["$prefix.k.weight"],
            tensors["$prefix.v.weight"];
            dims=1,
        )
        @assert size(w1) == (1536, 512, 1, 1)
        w1 = reshape(w1, (1536, 512))
        w1 = permutedims(w1, (2, 1))
        b1  = cat(
            tensors["$prefix.q.bias"],
            tensors["$prefix.k.bias"],
            tensors["$prefix.v.bias"],
            dims=1,
        )
        b1 = insertdims(b1, dims=1)
        w2 = tensors["$prefix.proj_out.weight"]
        @assert size(w2) == (512, 512, 1, 1)
        w2 = reshape(w2, (512, 512))
        w2 = permutedims(w2, (2, 1))
        b2 = tensors["$prefix.proj_out.bias"]
        b2 = insertdims(b2, dims=1)
        Dablock(g, t, w1, b1, w2, b2)
end

# Decoder group norm
struct Dgn
    g::Vector{Float32}
    t::Vector{Float32}
end

function get_dgn(tensors, prefix)
    g = tensors["$prefix.weight"]
    t = tensors["$prefix.bias"]
    Dgn(g, t)
end

# Decoder model
struct Dmodel
    dconv_pq::Dconv
    dconv_in::Dconv

    drblock_mid1::Drblock
    dablock::Dablock
    drblock_mid2::Drblock

    drblock_30::Drblock
    drblock_31::Drblock
    drblock_32::Drblock
    dconv_3::Dconv

    drblock_20::Drblock
    drblock_21::Drblock
    drblock_22::Drblock
    dconv_2::Dconv

    drcblock_10::Drcblock
    drblock_11::Drblock
    drblock_12::Drblock
    dconv_1::Dconv

    drcblock_00::Drcblock
    drblock_01::Drblock
    drblock_02::Drblock

    dgn::Dgn
    dconv_out::Dconv
end

function get_dmodel(tensors)
    dconv_pq = get_dconv(tensors, "first_stage_model.post_quant_conv")
    dconv_in = get_dconv(tensors, "first_stage_model.decoder.conv_in")

    drblock_mid1 = get_drblock(tensors, "first_stage_model.decoder.mid.block_1")
    dablock = get_dablock(tensors, "first_stage_model.decoder.mid.attn_1")
    drblock_mid2 = get_drblock(tensors, "first_stage_model.decoder.mid.block_2")

    drblock_30 = get_drblock(tensors, "first_stage_model.decoder.up.3.block.0")
    drblock_31 = get_drblock(tensors, "first_stage_model.decoder.up.3.block.1")
    drblock_32 = get_drblock(tensors, "first_stage_model.decoder.up.3.block.2")
    dconv_3 = get_dconv(tensors, "first_stage_model.decoder.up.3.upsample.conv")

    drblock_20 = get_drblock(tensors, "first_stage_model.decoder.up.2.block.0")
    drblock_21 = get_drblock(tensors, "first_stage_model.decoder.up.2.block.1")
    drblock_22 = get_drblock(tensors, "first_stage_model.decoder.up.2.block.2")
    dconv_2 = get_dconv(tensors, "first_stage_model.decoder.up.2.upsample.conv")

    drcblock_10 = get_drcblock(tensors, "first_stage_model.decoder.up.1.block.0")
    drblock_11 = get_drblock(tensors, "first_stage_model.decoder.up.1.block.1")
    drblock_12 = get_drblock(tensors, "first_stage_model.decoder.up.1.block.2")
    dconv_1 = get_dconv(tensors, "first_stage_model.decoder.up.1.upsample.conv")

    drcblock_00 = get_drcblock(tensors, "first_stage_model.decoder.up.0.block.0")
    drblock_01 = get_drblock(tensors, "first_stage_model.decoder.up.0.block.1")
    drblock_02 = get_drblock(tensors, "first_stage_model.decoder.up.0.block.2")

    dgn = get_dgn(tensors, "first_stage_model.decoder.norm_out")
    dconv_out = get_dconv(tensors, "first_stage_model.decoder.conv_out")

    Dmodel(
        dconv_pq,
        dconv_in,

        drblock_mid1,
        dablock,
        drblock_mid2,

        drblock_30,
        drblock_31,
        drblock_32,
        dconv_3,

        drblock_20,
        drblock_21,
        drblock_22,
        dconv_2,

        drcblock_10,
        drblock_11,
        drblock_12,
        dconv_1,

        drcblock_00,
        drblock_01,
        drblock_02,

        dgn,
        dconv_out,
    )
end

# Diffusion convolution
struct Fconv
    wc::Array{Float32, 4}  # [x, y, i, o]
    bc::Array{Float32, 4}  # [x, y, o, n]
end

function get_fconv(tensors, prefix)
    wc = tensors["$prefix.weight"]  # [o, i, y, x]
    wc = permutedims(wc, (4, 3, 2, 1))  # [x, y, i, o]
    bc = tensors["$prefix.bias"]  # [o]
    bc = insertdims(bc, dims=(1, 2, 4))  # [x, y, o, n]
    Fconv(wc, bc)
end

# Diffusion residual block
struct Frblock
    g1::Vector{Float32}
    t1::Vector{Float32}
    wc1::Array{Float32, 4}
    bc1::Array{Float32, 4}
    w::Matrix{Float32}
    b::Vector{Float32}
    g2::Vector{Float32}
    t2::Vector{Float32}
    wc2::Array{Float32, 4}
    bc2::Array{Float32, 4}
end

# Diffusion residual block w/ additional convolution
struct Frcblock
    g1::Vector{Float32}
    t1::Vector{Float32}
    wc1::Array{Float32, 4}
    bc1::Array{Float32, 4}
    w::Matrix{Float32}
    b::Vector{Float32}
    g2::Vector{Float32}
    t2::Vector{Float32}
    wc2::Array{Float32, 4}
    bc2::Array{Float32, 4}
    wc3::Array{Float32, 4}
    bc3::Array{Float32, 4}
end

function get_frblock(tensors, prefix)
    g1  = tensors["$prefix.in_layers.0.weight"]
    t1  = tensors["$prefix.in_layers.0.bias"]
    wc1 = tensors["$prefix.in_layers.2.weight"]  # [o, i, y, x]
    wc1 = permutedims(wc1, (4, 3, 2, 1))  # [x, y, i, o]
    bc1 = tensors["$prefix.in_layers.2.bias"]
    bc1 = insertdims(bc1, dims=(1, 2, 4))  # [x, y, i, o]
    w   = tensors["$prefix.emb_layers.1.weight"]
    b   = tensors["$prefix.emb_layers.1.bias"]
    g2  = tensors["$prefix.out_layers.0.weight"]
    t2  = tensors["$prefix.out_layers.0.bias"]
    wc2 = tensors["$prefix.out_layers.3.weight"]  # [o, i, y, x]
    wc2 = permutedims(wc2, (4, 3, 2, 1))  # [x, y, i, o]
    bc2 = tensors["$prefix.out_layers.3.bias"]  # [x, y, o, n]
    bc2 = insertdims(bc2, dims=(1, 2, 4))  # [x, y, i, o]
    Frblock(g1, t1, wc1, bc1, w, b, g2, t2, wc2, bc2)
end

function get_frcblock(tensors, prefix)
    g1  = tensors["$prefix.in_layers.0.weight"]
    t1  = tensors["$prefix.in_layers.0.bias"]
    wc1 = tensors["$prefix.in_layers.2.weight"]  # [o, i, y, x]
    wc1 = permutedims(wc1, (4, 3, 2, 1))  # [x, y, i, o]
    bc1 = tensors["$prefix.in_layers.2.bias"]
    bc1 = insertdims(bc1, dims=(1, 2, 4))  # [x, y, i, o]
    w   = tensors["$prefix.emb_layers.1.weight"]
    b   = tensors["$prefix.emb_layers.1.bias"]
    g2  = tensors["$prefix.out_layers.0.weight"]
    t2  = tensors["$prefix.out_layers.0.bias"]
    wc2 = tensors["$prefix.out_layers.3.weight"]  # [o, i, y, x]
    wc2 = permutedims(wc2, (4, 3, 2, 1))  # [x, y, i, o]
    bc2 = tensors["$prefix.out_layers.3.bias"]  # [x, y, o, n]
    bc2 = insertdims(bc2, dims=(1, 2, 4))  # [x, y, i, o]
    wc3 = tensors["$prefix.skip_connection.weight"]  # [o, i, y, x]
    wc3 = permutedims(wc2, (4, 3, 2, 1))  # [x, y, i, o]
    bc3 = tensors["$prefix.skip_connection.bias"]  # [x, y, o, n]
    bc3 = insertdims(bc3, dims=(1, 2, 4))  # [x, y, i, o]
    Frcblock(g1, t1, wc1, bc1, w, b, g2, t2, wc2, bc2, wc3, bc3)
end

# diffusion model
struct Fmodel
    time_w1::Matrix{Float32}
    time_b1::Vector{Float32}
    time_w2::Matrix{Float32}
    time_b2::Vector{Float32}
    fconv_i0::Fconv
    fconv_i3::Fconv
    fconv_i6::Fconv
    fconv_i9::Fconv
    fconv_o2::Fconv
    fconv_o5::Fconv
    fconv_o8::Fconv
    frblock_i1::Frblock
    frblock_i2::Frblock
    frblock_i5::Frblock
    frblock_i8::Frblock
    frblock_i10::Frblock
    frblock_i11::Frblock
    frblock_m0::Frblock
    frblock_m2::Frblock
    frcblock_i4::Frcblock
    frcblock_i7::Frcblock
    frcblock_o0::Frcblock
    frcblock_o1::Frcblock
    frcblock_o2::Frcblock
    frcblock_o3::Frcblock
    frcblock_o4::Frcblock
    frcblock_o5::Frcblock
    frcblock_o6::Frcblock
    frcblock_o7::Frcblock
    frcblock_o8::Frcblock
    frcblock_o9::Frcblock
    frcblock_o10::Frcblock
    frcblock_o11::Frcblock
end

function get_fmodel(tensors)
    time_w1 = tensors["model.diffusion_model.time_embed.0.weight"]  # [f₂, f₁]
    time_b1 = tensors["model.diffusion_model.time_embed.0.bias"]  # [f₂]
    time_w2 = tensors["model.diffusion_model.time_embed.2.weight"]  # [f₃, f₂]
    time_b2 = tensors["model.diffusion_model.time_embed.2.bias"]  # [f₃]
    fconv_i0 = get_fconv(tensors, "model.diffusion_model.input_blocks.0.0")
    fconv_i3 = get_fconv(tensors, "model.diffusion_model.input_blocks.3.0.op")
    fconv_i6 = get_fconv(tensors, "model.diffusion_model.input_blocks.6.0.op")
    fconv_i9 = get_fconv(tensors, "model.diffusion_model.input_blocks.9.0.op")
    fconv_o2 = get_fconv(tensors, "model.diffusion_model.output_blocks.2.1.conv")
    fconv_o5 = get_fconv(tensors, "model.diffusion_model.output_blocks.5.2.conv")
    fconv_o8 = get_fconv(tensors, "model.diffusion_model.output_blocks.8.2.conv")
    frblock_i1  = get_frblock(tensors, "model.diffusion_model.input_blocks.1.0")
    frblock_i2  = get_frblock(tensors, "model.diffusion_model.input_blocks.2.0")
    frblock_i5  = get_frblock(tensors, "model.diffusion_model.input_blocks.5.0")
    frblock_i8  = get_frblock(tensors, "model.diffusion_model.input_blocks.8.0")
    frblock_i10 = get_frblock(tensors, "model.diffusion_model.input_blocks.10.0")
    frblock_i11 = get_frblock(tensors, "model.diffusion_model.input_blocks.11.0")
    frblock_m0  = get_frblock(tensors, "model.diffusion_model.middle_block.0")
    frblock_m2  = get_frblock(tensors, "model.diffusion_model.middle_block.2")
    frcblock_i4  = get_frcblock(tensors, "model.diffusion_model.input_blocks.4.0")
    frcblock_i7  = get_frcblock(tensors, "model.diffusion_model.input_blocks.7.0")
    frcblock_o0  = get_frcblock(tensors, "model.diffusion_model.output_blocks.0.0")
    frcblock_o1  = get_frcblock(tensors, "model.diffusion_model.output_blocks.1.0")
    frcblock_o2  = get_frcblock(tensors, "model.diffusion_model.output_blocks.2.0")
    frcblock_o3  = get_frcblock(tensors, "model.diffusion_model.output_blocks.3.0")
    frcblock_o4  = get_frcblock(tensors, "model.diffusion_model.output_blocks.4.0")
    frcblock_o5  = get_frcblock(tensors, "model.diffusion_model.output_blocks.5.0")
    frcblock_o6  = get_frcblock(tensors, "model.diffusion_model.output_blocks.6.0")
    frcblock_o7  = get_frcblock(tensors, "model.diffusion_model.output_blocks.7.0")
    frcblock_o8  = get_frcblock(tensors, "model.diffusion_model.output_blocks.8.0")
    frcblock_o9  = get_frcblock(tensors, "model.diffusion_model.output_blocks.9.0")
    frcblock_o10 = get_frcblock(tensors, "model.diffusion_model.output_blocks.10.0")
    frcblock_o11 = get_frcblock(tensors, "model.diffusion_model.output_blocks.11.0")
    Fmodel(
        time_w1, time_b1, time_w2, time_b2,
        fconv_i0,
        fconv_i3,
        fconv_i6,
        fconv_i9,
        fconv_o2,
        fconv_o5,
        fconv_o8,
        frblock_i1,  
        frblock_i2,  
        frblock_i5,  
        frblock_i8,  
        frblock_i10, 
        frblock_i11, 
        frblock_m0,  
        frblock_m2,  
        frcblock_i4,  
        frcblock_i7,  
        frcblock_o0,  
        frcblock_o1,  
        frcblock_o2,  
        frcblock_o3,  
        frcblock_o4,  
        frcblock_o5,  
        frcblock_o6,  
        frcblock_o7,  
        frcblock_o8,  
        frcblock_o9,  
        frcblock_o10, 
        frcblock_o11, 
    )
end

end
