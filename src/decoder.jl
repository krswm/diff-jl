using SafeTensors

decref = load_safetensors("../../../Downloads/decref.safetensors")

function tshow(x)
    show(IOContext(stdout, :limit => true), "text/plain", x)
    println()
end

x = decref["l"]
@assert size(x) == (1, 4, 64, 64)
x ./= 0.18215
@assert size(x) == (1, 4, 64, 64)
tshow(x)
