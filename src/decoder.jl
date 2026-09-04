using JSON
using SafeTensors

include("model.jl")
using .Model

decref = load_safetensors("../../../Downloads/decref.safetensors")

tensors = load_safetensors("../../../Downloads/sd/v1-5/model.safetensors")
config = JSON.parsefile("../../../Downloads/sd/v1-5/config.json")
model = get_model(tensors, config)
keys(model.dconvs) |> println
exit(0)

function tshow(x)
    show(IOContext(stdout, :limit => true), "text/plain", x)
    println()
end

x = decref["l"]
@assert size(x) == (1, 4, 64, 64)
x ./= 0.18215
@assert size(x) == (1, 4, 64, 64)
