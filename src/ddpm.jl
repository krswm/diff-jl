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

module DDPM

export ddpm_step

using SafeTensors

#=
# Reference tensors.
rand42 = load_safetensors("../../../Downloads/rand42.safetensors")
=#

βs = range(√0.00085f0, √0.0120f0, 1000) .^ 2
αs = 1 .- βs
cumprod_αs = cumprod(αs)

function ddpm_step(curr_t, prev_t, xₜ, ϵ)
    # Julia is 1-based. scheduler's t is 0-based.
    curr_α_bar = cumprod_αs[curr_t + 1]
    prev_α_bar = prev_t ≥ 0 ? cumprod_αs[prev_t + 1] : 1.0f0
    αₜ = curr_α_bar / prev_α_bar

    # References: "Denoising Diffusion Probabilistic Models"
    # https://arxiv.org/pdf/2006.11239

    x₀ = (xₜ .- √(1 - curr_α_bar) * ϵ) ./ √curr_α_bar
    μₜ = √prev_α_bar * (1 - αₜ) / (1 - curr_α_bar) * x₀ + √αₜ * (1 - prev_α_bar) / (1 - curr_α_bar) * xₜ
    if curr_t > 0
        #=
        noise = rand42["n$curr_t"]  # TODO: Use random generator later.
        noise = permutedims(noise, (4, 3, 2, 1))
        =#
        noise = randn(Float32, 64, 64, 4, 1)
        variance = (1 - prev_α_bar) / (1 - curr_α_bar) * (1 - αₜ)
        result = μₜ + √variance * noise
    else
        result = μₜ
    end
    result
end

end
