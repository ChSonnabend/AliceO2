This is some documentation for the full system test setup.

If you just want to test a small dataset, you can skip the following steps, and jusddt skip to the end, where you will find a download with a prepared data set!

# Requirements:
- The FST needs a lot of memory. Please check the comments below, make sure your system has enough memory, and change the memory sizes in the command lines accordingly.
- ulimits: The FST needs large ulimits for memory and virtual memory (`ulimit -m` / `ulimit -v`). This is usually no problem since they are usually unlimited. If GPUs are used, the FST also needs `ulimit -l` (for locked memory) unlimited, which is usualy not the system default. Finally, if data is replayed from raw files (not with DataDistribution), the FST will open many files, and `ulimit -n` should be at least 4096. Note that in most distributions the hard ulimits are configured in `/etc/security/limits.conf`.
- The FST needs to access the CCDB. For this, you should run the FST with an alien token. Alternatively, if you are on the EPN you can use the EPN-internal CCDB server by exporting `ALL_EXTRA_CONFIG="NameConf.mCCDBServer=http://o2-ccdb.internal;"` and by setting the DPL CCDB backend on the command line. If you are using `start-tmux.sh` for the 8 GPU FST, the CCDB backends are set automatically.

# Creating the raw data and run the FST:
1. First some remarks on the number of events and the memory size:
    - Generation (simulation) of the full time frame with ~550 collisions will need ~256 GB, processing will take less.
    - Due to the sampling of the bunch crossings, the exact number of collissions that will be in the TF is not clear, thus one should simulate 600 collisions to generate a full 128 orbit TF.
    - The default shared memory size is 2 GB, and must be increased significantly for large time frames, 128 GB is sufficient for 128 orbit TF, 160 GB is needed if MC labels are present in addition.
    - The GPU memory allocation should be set to ~13 GB for 70 orbits and 21 GB for 128 orbits.
    - I'd suggest to do a first small test with 1-5 events to check the machinery, 100 events is already a good size which should not exhaust the memory, I'd go to 600 only after 100 works.
1. Compile O2 with GPU support, in addition you need O2sim, DataDistribution, and Readout (latest versions from alidist will do).
 GPUs for O2 should be auto-detected, but you can set the environment variables ALIBUILD_ENABLE_CUDA / ALIBUILD_ENABLE_HIP to enforce it (and get a failure when detection fails). Look for CMake log messages "Building GPUTracking with CUDA support" (etc) to verify.
 For more information, see https://github.com/AliceO2Group/AliceO2/blob/dev/GPU/documentation/build-O2.md
1. Optionally place some binary configuration files in the simulation folder. Default objects will be used if no such files are placed. There are instructions at the end of this post how to generate these files. (Currently, these files are: matbud.root, ITSdictionary.bin, ctf_dictionary.root, tpctransform.root, dedxsplines.root, and tpcpadgaincalib.root)
1. Load the O2sim environment (`alienv enter O2sim/latest`) and run the following full system test script for a full simulation and digits to raw conversion (this will already include 1 CPU reconstruction run):
    ```
    NEvents=600 NEventsQED=35000 SHMSIZE=128000000000 TPCTRACKERSCRATCHMEMORY=30000000000 $O2_ROOT/prodtests/full_system_test.sh
    ```
    - This create a full 128 orbit TF with 550 collisions and uses 35000 interactions for the QED background
    - It uses 128 GB of shared memory
    - The scratch memory size for the TPC reconstruction is set to 24 GB (Note, this is the CPU-equivalent of the GPU memory size, since this phase will only run on the CPU).
1. Test of the workflow using the raw-file-reader: Run the so far largest workflow, The GPU and SHM memory sizes must be reasonably large (see above).
    ```
    SHMSIZE=128000000000 NTIMEFRAMES=10 TFDELAY=100 GPUTYPE=CPU $O2_ROOT/prodtests/full-system-test/dpl-workflow.sh
    ```
    Note that This uses 128 GB of SHM, runs only on the CPU, and processes the time frame 10 times in a loop with 100 s delay between the publiushing.
    - For a documentation of the options, see https://github.com/AliceO2Group/AliceO2/blob/dev/prodtests/full-system-test/documentation/full-system-test.md
    - For running on the GPU (4 GPUs with the HIP backend), please do
        ```
        SHMSIZE=128000000000 NTIMEFRAMES=10 TFDELAY=10 GPUTYPE=HIP NGPUS=4 GPUMEMSIZE=22000000000 $O2_ROOT/prodtests/full-system-test/dpl-workflow.sh
        ```
This will use 4 GPU with the HIP backend and allocate 22 GB of scratch memory on the GPU (should be sufficient for 128 orbit TF). You can change the GPU type as indicated in the linked README.md above, e.g. `GPUTYPE=CUDA NGPUS=1` for 1 CUDA GPU.
1. With this, the full chain is running inside O2 DPL. Next we are adding DataDistribution.
    1. Ceate the TF files as explained in the subtask ([raw-tf-conversion.md](https://github.com/AliceO2Group/AliceO2/blob/dev/prodtests/full-system-test/documentation/raw-tf-conversion.md)). For convenience, there is a script that should do it automatically, from a shell that has loaded both DataDistribution and Readout: `$O2_ROOT/prodtests/full-system-test/convert-raw-to-tf-file.sh`.
    1. Enter the O2 environment, and run the following script (please adjust the variables as in the test before).
        ```
        EXTINPUT=1 SHMSIZE=128000000000 GPUTYPE=CPU $O2_ROOT/prodtests/full-system-test/dpl-workflow.sh
        ```
        - As a first optional test without DataDistribution, we can take the RawReader to feed the data in the way DataDistribution does. Run the following script in a second shell within the O2 environment. (Please adjust the variables as noted above)
            ```
            SHMSIZE=128000000000 NTIMEFRAMES=10 TFDELAY=100 $O2_ROOT/prodtests/full-system-test/raw-reader.sh
            ```
    1. In a second shell with DataDistribution, run the following script (adjust the 2 variables for memory size as needed for your data, and set the TF_DIR variable to the folder where you recorded the time frame). Make sure you start this script ONLY AFTER the DPL workflow has fully started! There is no number of timeframes, it will run in an endless loop
        ```
        SHMSIZE=128000000000 DDSHMSIZE=32000 TFDELAY=100 $O2_ROOT/prodtests/full-system-test/datadistribution.sh
        ```
1. The full chain that will be running on the EPN farm is a bit more complicated. It consists of:
    - 2 instances of the dpl-workflow driving 4 GPUs each, one per NUMA domain.
    - 1 instance of data distribution feeding a shared input buffer.
        The following script runs the full system test in the 8 GPU EPN configuration using tmux with 3 sessions:{code}TFDELAY=2.8457 NTIMEFRAMES=128 $O2_ROOT/prodtests/full-system-test/start-tmux.sh dd{code}
    - Note that number of GPUs / memory sizes are automatically set by start-tmux.sh.
    - This TFDELAY is the rate for processing 1/250th of 50 kHz Pb-Pb with average time frames. Since the occupancy of your simulated timeframe will fluctuate, it is suggested to scale the TFDELAY linearly with the number of tpc clusters (shown in the console output of the dpl-workflow), with the average corresponding to 2.8457 s being 313028012 clusters.
    - You can for testing alternatively use the rawreader instead of datadistribution as input in the start_tmux.sh script by passing rr instead of dd.
1. On the EPN, an SHM management tool owns the memory in the background and keeps it locked. This is done in order to speed up the startup. This behavior can be reproduced in the full system test, by setting the env variable `SHM_MANAGER_SHMID` to the shm id to be used (must be set for both `start_tmux.sh` and `shm-tool.sh`) you can juse use `SHM_MANAGER_SHMID=1` for a test) and running in a separate shell before starting `start_tmux.sh`
    ```
    SHM_MANAGER_SHMID=1 SHMSIZE=$((128<<30)) DDSHMSIZE=$((128<<10)) $O2_ROOT/prodtests/full-system-test/shm-tool.sh
    SHM_MANAGER_SHMID=1 TFDELAY=2.8457 NTIMEFRAMES=8 $O2_ROOT/prodtests/full-system-test/start-tmux.sh dd
    ```

Note that as of 2024, we are running with a TF length of 32 orbits.
To simulate time frames accordingly, please set `NEVENTS=200 NHBPERTF=32` for full-system-test.sh.

---

# Remarks for running with distortions:
1. To run the digitization with distortions, add the following to the digitizer command (using map inputSCDensity3D_8000_0 from file../InputSCDensityHistograms_8000events.root):
    ```
    --distortionType 2 --initialSpaceChargeDensity=../InputSCDensityHistograms_8000events.root,inputSCDensity3D_8000_0
    ```
1. To rerun the digitization with the same BC sampling for the collisions add
    ```
    --incontext collisioncontext.root
    ```
1. To create the tpc fast transform map from the SCD object run:
    ```
    root -l -q -b ~/alice/O2/Detectors/TPC/reconstruction/macro/createTPCSpaceChargeCorrection.C++'("../InputSCDensityHistograms_8000events.root", "inputSCDensity3D_8000_0")'
    ```
1. In order to use the fast transform map for TPC tracking, add to the tpc-recop-workflow:
    ```
    --configKeyValues "GPU_global.transformationFile=tpctransform.root"
    ```

---

# Remarks for creating other prerequisite binary files:
1. To create the CTF dictionary: Run the full system test workflow once setting the env variable CREATECTFDICT=1:
    ```
    CREATECTFDICT=1 $O2_ROOT/prodtests/full-system-test/dpl-workflow.sh
    ```
1. Create the ITS pattern dictionary
    ```
    o2-its-reco-workflow --trackerCA --disable-mc --configKeyValues "fastMultConfig.cutMultClusLow=30000;fastMultConfig.cutMultClusHigh=2000000;fastMultConfig.cutMultVtxHigh=500"
    root -b -q ~/alice/O2/Detectors/ITSMFT/ITS/macros/test/CheckTopologies.C++
    ```
    - Note that the ITS dictionary used for raw generation and for reconstruction must be the same. I.e., if you change this, you have to either restart from scratch with the new dictionary file or rerun the ITS raw generation part of `$O2_ROOT/prodtests/full_system_test.sh`.
1. To create the material lookup table
    ```
    root -l -q -b $O2_ROOT/Detectors/Base/test/buildMatBudLUT.C
    ```
1. missing here: dedxsplines.root, tpcpadgaincalib.root

---

# Measuring startup time:
- In order to measure the time for each individual GPU memory registration step, please add `CONFIG_EXTRA_PROCESS_o2_gpu_reco_workflow="GPU_global.benchmarkMemoryRegistration=1;"`. This should show you 2 times ~2 seconds per GPU process for the 2 large segments (DD and the global segment, could also report some additional smaller segments, only 1 in case you don't use the readout proxy).
- In order to measure the total startup time, you can use the `start_tmux.sh` script with the option `FST_BENCHMARK_STARTUP=1`. It will print for both DPL chains 2 times at the beginning: The first is when it starts the workflow JSON generation, the second is after the JSON generation when the actual workflow is started. For the process startup time, you have to take the difference from that time until the time when the last process has reched the READY state. (Note that this should be done with the `$O2_ROOT/prodtests/full-system-test/shm-tool.sh` as instructed above.)
    ```
    Fri Jan 28 11:25:48 CET 2022
    Fri Jan 28 11:25:56 CET 2022
    [...]
    [1456583:gpu-reconstruction_t0]: [11:26:18][INFO] fair::mq::Device running...
    ```
    - This corresponds to a JSON creation time of 8 seconds (will usually not cound for the startup since it is cached, and a process startup time of 22 seconds.
---

# Other remarks:# Other remarks:
1. To run with low b-field, add to o2-sim:
    ```
    --field -2
    ```
1. To create a sample of multiple TF files for StfBuilder, use the script `$O2_ROOT/prodtests/full-system-test/generate_timeframe_files.sh`.
