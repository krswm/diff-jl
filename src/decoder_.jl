using JSON
using SafeTensors

include("model.jl")
using .Model

function tshow(x)
    println(size(x))
    show(IOContext(stdout, :limit => true), "text/plain", x)
    println()
end

function decode(x)
    @assert size(x) == (1, 4, 64, 64)
    @assert eltype(x) == Float32
    x = reshape(x, (4, 64, 64))  # [o, y, x]
    x = permutedims(x, (3, 2, 1))  # [x, y, o]
    @assert size(x) == (64, 64, 4)
    @assert eltype(x) == Float32
    x ./= 0.18215f0  # [x, y, o]
    x |> tshow
end

decref = load_safetensors("../../../Downloads/decref.safetensors")
model = begin
    tensors = load_safetensors("../../../Downloads/sd/v1-5/model.safetensors")
    config = JSON.parsefile("../../../Downloads/sd/v1-5/config.json")
    get_model(tensors, config)
end

x = decref["l"]
decode(x)
