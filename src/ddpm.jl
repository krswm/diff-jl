using SafeTensors

# Reference tensors.
rand42 = load_safetensors("../../../Downloads/rand42.safetensors")
ddpmref = load_safetensors("../../../Downloads/ddpmref.safetensors")

curr_t = 900
prev_t = 800
