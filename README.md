# Basic CUDA PROJECTS


Consists of 2 CUDA Projects:-

1. Vector Addition: Adds two simple vectors on the GPU. I used this to understand how indexing works in GPU thread blocks.
2. Image to Grayscale: Converts in a single image to grayscale.

## Interesting Statistics about the Programs

For the Vector Addition

<img width="1083" alt="image" src="https://github.com/user-attachments/assets/d64b5e3c-0208-4d16-85b4-1b1688b4b61f">


- We allocate 16 MB of memory for the program (indicated by the Dark Green Bar)
- Kernel Execution: Represents kernel execution on the GPU, which refers to the actual code or function running on the GPU cores.
- The value "2%" likely indicates that this timeline snapshot is capturing a segment where only 2% of kernel functions are active.
- On the 813ms we see an increase in the memory usage. It's a gradual increase in the memory allocation which allocates each vector to the GPU. We allocated 16 MB.
- Each increase indicates each vector being allocated
- The GPU is using 98% (of 16MB) of its allocated memory resources.
- The light green bars indicate that memory is being allocated.
- The dark red bars indicate that memory is being used

For the Gray Scale Image

<img width="1217" alt="image" src="https://github.com/user-attachments/assets/52eb06f8-588f-450e-ab48-871d22008672">

- In this case, since we are dealing with image data, we need to allocate 64 MBs instead. (indicated by the Dark Green Bar)
- Instead of using multiple vectors, we loaded all the image data. This is why we see an immediate increase instead of a gradual increadse
- We are also able to see a more defined period during which the function is ran, which is indicated by the purple block in kernel.
- This is because it takes a longer time to run compared to adding two vectors

<img width="1014" alt="image" src="https://github.com/user-attachments/assets/bb91af63-da14-4c42-aa2e-cba83e3f8bb1">

- The reason why we see the memory being allocated after a long period of time is because we deallocate only after the program is done.
- We only free the memory after the program ends.
- We could free the memory after the CUDA function is ran and the data is transferred to the CPU.
- This would result in the memory usage going to 0 immediately after red.
