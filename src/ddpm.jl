using SafeTensors

# Reference tensors.
rand42 = load_safetensors("../../../Downloads/rand42.safetensors")
ddpmref = load_safetensors("../../../Downloads/ddpmref.safetensors")

curr_t = 900
prev_t = 800

βs = range(√0.00085f0, √0.0120f0, 1000) .^ 2
αs = 1 .- βs
cumprod_αs = cumprod(αs)

# Julia is 1-based. scheduler's t is 0-based.
curr_α_bar = cumprod_αs[curr_t + 1]
prev_α_bar = prev_t ≥ 0 ? cumprod_αs[prev_t + 1] : 0.0f0
αₜ = curr_α_bar / prev_α_bar

xₜ = ddpmref["l"]
ϵ = ddpmref["o"]

# References: "Denoising Diffusion Probabilistic Models"
# https://arxiv.org/pdf/2006.11239

x₀ = (xₜ .- √(1 - curr_α_bar) * ϵ) ./ √curr_α_bar




μₜ = √prev_α_bar * (1 - αₜ) / (1 - curr_α_bar) * x₀ + √αₜ * (1 - prev_α_bar) / (1 - curr_α_bar) * xₜ

noise = rand42["n$curr_t"]

variance = (1 - prev_α_bar) / (1 - curr_α_bar) * (1 - αₜ)

result = μₜ + √variance * noise

show(IOContext(stdout, :limit => true), "text/plain", result)

# Done!
