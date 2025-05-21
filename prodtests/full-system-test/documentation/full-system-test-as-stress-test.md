This is a quick summary how to run the full system test (FST) as stress test on the EPN. (For the full FST documentation, see https://github.com/AliceO2Group/AliceO2/blob/dev/prodtests/full-system-test/documentation/full-system-test-setup.md and https://github.com/AliceO2Group/AliceO2/blob/dev/prodtests/full-system-test/documentation/full-system-test.md)

# Preparing the data set
- I usually try to keep an up-to-date data set that can be used in `/home/drohr/alitest/tmp-fst*`. The folder with the highest number is the latest dataset. However, data formats are still evolving, and it requires rerunning the simulation regularly. I.e. please try my latest data set, if it doesn't work, please generate a new one as described below.
- Short overview how to generate a FST Pb-Pb 128 orbit data set:
  - The O2 binaries installed on the EPN via RPMs use the `o2-dataflow` defaults and cannot run the simulation, and also they lack readout. Thus you need to build `O2PDPSuite` and `Readout` (the version matching the O2PDPSuite RPM you want to use for running the test) yourself with `alibuild` on an EPN: `aliBuild --defaults o2 build O2PDPSuite Readout --jobs 32 --debug`. The flag `--jobs` configures the number of parallel jobs and can be changed.
  - Enter the O2PDPSuite environment either vie `alienv enter O2PDPSuite/latest Readout/latest`.
  - Go to an empty directory.
  - Run the FST simulation via: `NEvents=650 NEventsQED=10000 SHMSIZE=128000000000 TPCTRACKERSCRATCHMEMORY=40000000000 SPLITTRDDIGI=0 GENERATE_ITSMFT_DICTIONARIES=1 $O2_ROOT/prodtests/full_system_test.sh`
  - Material budget table (e.g. from here https://alice.its.cern.ch/jira/browse/O2-2288) now comes from CCDB, no need any more to pull it manually.
  - Create a timeframe file from the raw files: `$O2_ROOT/prodtests/full-system-test/convert-raw-to-tf-file.sh`.
  - Prepare the ramdisk folder: `mv raw/timeframe raw/timeframe-org; mkdir raw/timeframe-tmpfs; ln -s timeframe-tmpfs raw/timeframe`

# Running the full system test
- Enter the environment! On an EPN do `module load O2PDPSuite` (this will load the latest O2 software installed on that EPN).
- Go into the folder with the data set (you might need to create one, see above).
- Prepare the ramdisk with the data: `sudo mount -t tmpfs tmpfs raw/timeframe-tmpfs; sudo cp raw/timeframe-org/* raw/timeframe`
  - (NOTE that the ramdisk might already be present from previous tests, or in a different folder. Check the mounted tmpfs filesystems (`mount | grep tmpfs`), and don't mount multiple of them since memory is critical!)
  - If you do not have root permissions and cannot create a ramdisk, the test will also work without. In that case you should decrease the publishing rate below to `TFDELAY=5`.
- Make sure disk caches are cleared: as ROOT do: `echo 1 > /proc/sys/vm/drop_caches`
- In order to run the Full System Test, the workflow must be able to access the CCDB. Normally, if you run as user, you must make sure to have an alien token present. On the EPN, one can use the EPN-internal CCDB server instead, which does not require alien access. If you use the `start-tmux.sh`, the env variables are set automatically to access the EPN-internal CCDB server.
- Start the FST with 2 NUMA domains: `TFDELAY=2.5 NTIMEFRAMES=1000000 $O2_ROOT/prodtests/full-system-test/start_tmux.sh dd`

This will start a tmux session with 3 shells, the upper 2 shells are the 2 DPL workflows, one per NUMA domain, for the processing. The lower shell is the input with DataDistribution's StfBuilder. Leave it running and check that the StfBuilder doesn't complain that its buffer is full. Then the EPN can sustain the rate.

# **NOTE**
- Attached to this ticket is a screenshot of how the console should look like:
  - The DD console (on the bottom) should not show warnings about full buffers.
  - The other 2 consoles (1 per NUMA domain) should show the processing times per TF for the GPU reconstruction:
    ```
    [2974450:gpu-reconstruction_t3]: [10:50:38][INFO] GPU Reoncstruction time for this TF 26.77 s (cpu), 17.8823 s (wall)
    ```
    This should be 17 to 18 seconds, and you should see it for all 4 GPUs on both NUMA domains (`reconstruction_t0` to `reconstruction_t3`)
