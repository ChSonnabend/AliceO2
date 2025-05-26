This ticket will serve as documentation how to enable which GPU features and collect related issues.

So far, the following features exist:
 * GPU Tracking with CUDA
 * GPU Tracking with HIP
 * GPU Tracking with OpenCL (>= 2.1)
 * OpenGL visualization of the tracking
 * ITS GPU tracking

GPU support should be detected and enabled automatically.
If you just want to reproduce the GPU build locally without running it, it might be easiest to use the GPU CI container (see below).
The provisioning script of the container also demonstrates which patches need to be applied such that everything works correctly.

*GPU Tracking with CUDA*
 * The CMake option `-DENABLE_CUDA=ON/OFF/AUTO` steers whether CUDA is forced enabled / unconditionally disabled / auto-detected.
 * The CMake option `-DCUDA_COMPUTETARGET=...` fixes a GPU target, e.g. 61 for PASCAL or 75 for Turing (if unset, it compiles for the lowest supported architecture)
 * CUDA is detected via the CMake language feature, so essentially nvcc must be in the Path.
 * We require CUDA version >= 12.8
 * CMake will report "Building GPUTracking with CUDA support" when enabled.

*GPU Tracking with HIP*
 * HIP and HCC must be installed, and CMake must be able to detect HIP via find_package(hip).
 * If HIP and HCC are not installed to /opt/rocm, the environment variables `$HIP_PATH` and `$HCC_HOME` must point to the installation directories.
 * HIP from ROCm >= 4.0 is required.
 * The CMake option `-DHIP_AMDGPUTARGET=...` forces a GPU target, e.g. gfx906 for Radeon VII (if unset, it auto-detects the GPU).
 * CMake will report "Building GPUTracking with HIP support" when enabled.
 * It may be that some patches must be applied to ROCm after the installation. You find the details in the provisioning script of the GPU CI container below.

*GPU Tracking with OpenCL (Needs Clang >= 18 for compilation)*
 * Needs OpenCL library with version >= 2.1, detectable via CMake find_package(OpenCL).
 * Needs the SPIR-V LLVM translator together with LLVM to create the SPIR-V binaries, also detectable via CMake.

*OpenGL visualization of TPC tracking*
 * Needs the following libraries (all detectable via CMake find_package): libOpenGL, libGLEW, libGLFW, libGLU.
 * OpenGL must be at least version 4.5, but this is not detectable at CMake time. If the supported OpenGL version is below, the display is not/partially built, and not available at runtime. (Whether it is not or partially built depends on whether the maximum OpenGL version supported by GLEW or that of the system runtime in insufficient.)
 * Note: If ROOT does not detect the system GLEW library, ROOT will install its own very outdated GLEW library, which will be insufficient for the display. Since the ROOT include path will come first in the order, this will prevent the display from being built.
 * CMake will report "Building GPU Event Display" when enabled.

*Vulkan visualization*
 * similar to OpenCL visualization, but with Vulkan.

*ITS GPU Tracking*
 * So far supports only CUDA and HIP, support for OpenCL might come.
 * The build is enabled when the "GPU Tracking with CUDA" (as explained above) detects CUDA, same for HIP.
 * CMake will report "Building ITS CUDA tracker" when enabled, same for HIP.

*Using the GPU CI container*
 * Setting up everything locally might be somewhat time-consuming, instead you can use the GPU CI cdocker container.
 * The docker images is `alisw/slc8-gpu-builder`.
 * The container exports the `ALIBUILD_O2_FORCE_GPU` env variable, which force-enables all GPU builds.
 * Note that it might not be possible out-of-the-box to run the GPU version from within the container. In case of HIP it should work when you forwards the necessary GPU devices in the container. For CUDA however, you would either need to (in addition to device forwarding) match the system CUDA driver and toolkit installation to the files present in the container, or you need to use the CUDA docker runtime, which is currently not installed in the container.
 * There are currently some patches needed to install all the GPU backends in a proper way and together. Please refer to the container provisioning script [provision.sh](https://github.com/alisw/docks/blob/master/slc9-gpu-builder/provision.sh). If you want to reproduce the installation locally, it is recommended to follow the steps from the script.

*Summary*

If you want to enforce the GPU builds on a system without GPU, please set the following CMake settings:
 * `ENABLE_CUDA=ON`
 * `ENABLE_HIP=ON`
 * `ENABLE_OPENCL=ON
 * `HIP_AMDGPUTARGET=default`
 * `CUDA_COMPUTETARGET=default`
Alternatively you can set the environment variables `ALIBUILD_ENABLE_CUDA=1` and `ALIBUILD_ENABLE_HIP=1` to enforce building CUDA or HIP without modifying the alidist scripts.
