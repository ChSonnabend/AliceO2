This ticket describes how to build the O2 GPU TPC Standalone benchmark (in its 2 build types), and how to run it.

The purpose of the standalone benchmark is to make the O2 GPU TPC reconstruction code available standalone. It provides
- external tests when people do not have / want to build O2, have no access to alien for CCDB, etc.
- fast standalone tests without running O2 workflows and overhead from CCTD.
- faster build times than rebuilding O2 for development.

# Compiling

The standalone benchmark is build as part of O2, and it can be built standalone.

As part of O2, it is available from the normal O2 build as the executable `o2-gpu-standalone-benchmark`, GPU support is available for all GPU types supported by the O2 build.

Building it as standalone benchmark requires several dependencies, and provides more control which features to enable / disable.
The dependencies can be taken from the system, or we can use alidist to build O2 and take the dependencies from there.

In order to do the latter, please execute:
```
cd ~/alice # or your alice folder
aliBuild build --defaults o2 O2
source O2/GPU/GPUTracking/Standalone/cmake/prepare.sh
```

Then, in order to compile the standalone tool, assuming to have it in ~/standalone and build in ~/standalone/build, please run:
```
mkdir -p ~/standalone/build
cd ~/standalone/build
cmake -DCMAKE_INSTALL_PREFIX=../ ~/alice/O2/GPU/GPUTracking/Standalone/
nano config.cmake # edit config file to enable / disable dependencies as needed. In case cmake failed, and you disabled the dependency, just rerun the above command.
make install -j32
```

You can edit certain build settings in `config.cmake`. Some of them are identical to the GPU build settings for O2, as described in [build-O2.md](https://github.com/AliceO2Group/AliceO2/blob/dev/GPU/documentation/build-O2.md).
And there are plenty of additional settings to enable/disable event display, qa, usage of ROOT, FMT, etc. libraries.

This will create the `ca` binary in `~/standalone`, which is basically the same as the `o2-gpu-standalone-benchmark`, but built outside of O2.

As an exacmple you can also have a look at [build.sh](https://github.com/AliceO2Group/AliceO2/blob/dev/GPU/GPUTracking/Standalone/cmake/build.sh), which is used by the CI.

# Running

The following command lines will use `./ca`, in case you use the executable from the O2 build, please replace by `o2-gpu-standalone-benchmark`.

You can get a list of command line options by `./ca --help` and `./ca --helpall`.

In order to run, you need a dataset. See the next section for how to create a dataset. Datasets are stored in `~/standalone/events`, and are identified by their folder names. The following commands assume a testdataset of name `o2-pbpb-100`.

To run on that data, the simpled command is `./ca -e o2-pbpb-100`. This will automatically use a GPU if available, trying all backends, otherwise fall back to CPU.
You can force using GPU or CPU with `-g` and `-c`.
You can select the backend via `--gpuType CUDA|HIP|OCL|OCL2`, and inside the backend you can select the device number, if multiple devices exist, via `--gpuDevice i`.

The flag `--debug` (-2 to 6) enables increasingly extensive debug output, and `--debug 6` stores full data dumpts of all intermediate steps to files.
>= `--debug 1` has a performance impact since it adds serialization points for debugging. For timing individual kernels, `--debug 1` prints timing information for all kernels.
An example line would .e.g. be
```
./ca -e o2-pbpb-100 -g --gpuType CUDA --gpuDevice 0 --debug 1
```

Some other noteworthy options are `--display` to run the GPU event display, `--qa` to run a QA task on MC data, `--runs` and `--runs2` to run multiple iterations of the benchmark, `--printSettings` to print all the settings that were used, `--memoryStat` to print memory statistics, `--sync` to run with settings for online reco, `--syncAsync` to run online reco first, and then offline reco on the produced TPC CTF data, `--setO2Settings` to use some defaults as they are in O2 not in the standalone version, `--PROCdoublePipeline` to enable the double-threaded pipeline for best performance (works only with multiple iterations, and not in async mode), and `--RTCenable` to enable the run time compilation improvements (check also `--RTCcacheOutput`).
With `--memSize` you can control the amount of GPU memory to use, and with `--inputMemory` and `--outputMemory` GPU-registered input/output buffers can be preallocated (as is the SHM memory when running in O2).
An example for a benchmark that runs with the same settings as in online data taking would be:
```
./ca -e o2-pbpb-100 -g --gpuType HIP --sync --setO2Settings --PROCdoublePipeline --RTCenable --runs 10 --memSize 15000000000 --inputMemory 6000000000 --outputMemory 10000000000
```

For setting a GPU device, you can use the `--gpuDevice` option with the GPU index.
For ROCm with many GPUs, however, like on the EPNs with 8 GPUs, it is better to set the `ROCR_VISIBLE_DEVICES` env variable to the GPU you want to use.
MAKE SURE TO CHECK IF IT IS ALREADY SET BY SLURM WHEN YOU GET THE NODE!!! IN THAT CASE, USE ONLY THE GPUS ASSIGNED TO YOU BY SLURM!

Finally, also NUMA pinning can play a role. On the EPN, you should use memory and GPUs and CPU cores from the same NUMA domain.
For a reaslistic benchmark using GPU 0 on the EPNs, please use:
```
ROCR_VISIBLE_DEVICES=0 numactl --membind 0 --cpunodebind 0 ./ca -e o2-pbpb-100 --gpuType HIP --memSize 15000000000 --inputMemory 6000000000 --outputMemory 10000000000 --sync --runs 10 --RTCenable --setO2Settings --PROCdoublePipeline
```

Note that on the MI50 nodes, we use only <16 GB of memory, since there is a performance regression when using the upper half of the 32 GB. In order to fit in the 16 GB, we have reduced the time frame length to 32 orbits from 2024 onwards.

# Generating a dataset

The standalone benchmark supports running on Run2 data exported from AliRoot, or to run on Run3 data from O2. This document covers only the O2 case.
In o2, `o2-tpc-reco-workflow` and the `o2-gpu-reco-workflow` can dump event data with the `configKeyValue` `GPU_global.dump=1;`.
This will dump the event data to the local folder, all dumped files have a `.dump` file extension. If there are multiple TFs/events processed, there will be multiple `event.i.dump` files. In order to create a standalone dataset out of these, just copy all the `.dump` files to a subfolder in `~/standalone/events/[FOLDERNAME]`.

Data can be dumped from raw data, or from MC data, e.g. generated by the Full System Test. In case of MC data, also MC labels are dumped, such that they are used in the `./ca --qa` mode.

To get a dump from simulated data, please run e.g. the FST simulation as described in [full-system-test-setup.md](https://github.com/AliceO2Group/AliceO2/blob/dev/prodtests/full-system-test/documentation/full-system-test-setup.md).
A simple run as
```
DISABLE_PROCESSING=1 NEvents=5 NEventsQED=100 SHMSIZE=16000000000 $O2_ROOT/prodtests/full_system_test.sh
```
should be enough.

Afterwards run the following command to dump the data:
```
SYNCMODE=1 CONFIG_EXTRA_PROCESS_o2_gpu_reco_workflow="GPU_global.dump=1;" WORKFLOW_DETECTORS=TPC SHMSIZE=16000000000 $O2_ROOT/prodtests/full-system-test/dpl-workflow.sh
```

To dump standalone data from CTF raw data in `myctf.root`, you can use the same script, e.g.:
```
CTFINPUT=1 INPUT_FILE_LIST=myctf.root CONFIG_EXTRA_PROCESS_o2_gpu_reco_workflow="GPU_global.dump=1;" WORKFLOW_DETECTORS=TPC SHMSIZE=16000000000 $O2_ROOT/prodtests/full-system-test/dpl-workflow.sh
```

On the EPNs, you can find some reference data sets at `/home/drohr/standalone/events`.
