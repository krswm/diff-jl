# Stable Diffusion Inference with Julia

I built a Stable Diffusion inference engine with Julia from scratch.

## Credits

- [*Stable Diffusion implemented from scratch in PyTorch*](https://github.com/hkproj/pytorch-stable-diffusion) and [its accompanying video](https://www.youtube.com/watch?v=ZBKpAp_6TGI) for teaching me the technical aspect of the model and providing me a reference implementation.
- [*Step by step explanation of 2D convolution implemented as matrix multiplication using toeplitz matrices*](https://github.com/alisaaalehi/convolution_as_multiplication) for teaching me how to calculate a 2D convolution using a matrix multiplication.
- [FluxML](https://fluxml.ai) for providing me `conv` from [NNlib.jl](https://github.com/FluxML/NNlib.jl) and `load_safetensors` from [SafeTensors.jl](https://github.com/FluxML/SafeTensors.jl).
- [Julia](https://julialang.org) for providing me an amazing programming language.

## Development

I was interested in the diffusion models in machine learning because of its application on science.
I decided to implement an inference engine of a diffusion model from scratch.
I chose Stable Diffusion as a kickstart of my diffusion engine implementation.
I targeted Stable Diffusion even though it is not a scientific application because its weigth is openly available and there already was many technical information available online.
I am planning to implement something scientific based on this project later.

- 2026-08-25: I started this project.
- 2026-09-09: I finished my implementation of Stable Diffusion inference engine.

I used [the weight of Stable diffusion](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) and [a reference implementation (*Stable Diffusion implemented from scratch in PyTorch*)](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) only for the purpose to observe their behavior as diffusion model.
Except for that, I did *not* use generative AI for this project at all.
