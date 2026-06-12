add_executable(smoke_kernel smoke_kernel.cu)

set_target_properties(smoke_kernel PROPERTIES
    CUDA_ARCHITECTURES 89
)

target_compile_options(smoke_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)