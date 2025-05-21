This procedure will create (S)TF files from raw data prepared as described in the main ticket. The data must be using RDHv6.
Create configuration for the readout.exe with all input files we want in the TF. This will create rdo_TF.cfg file.

 
```
ulimit -n 4096 # Make sure we can open sufficiently many files cd raw# ls raw: ITS TPC TOF ...

# copy gen_rdo_cfg.sh script attached here to the raw directory
# Run the script with number of HBF/TF and list directories you want to include in the TF

~raw> ./gen_rdo_cfg.sh 128 TPC ITS TOF # ... others{code}
``` 

In a separate shell load a recent DataDistribution module and start StfBuilder to record the TF:
```
export TF_PATH=$(pwd)
StfBuilder --id=stfb --detector-rdh=6 --detector-subspec=feeid --stand-alone  --channel-config "name=readout,type=pull,method=connect,address=ipc:///tmp/readout-to-datadist-0,transport=shmem,rateLogging=1" --data-sink-dir=${TF_PATH} --data-sink-sidecar --data-sink-enable
```

Start the readout.exe (at least v1.4.3) using the generated config file. The dataflow will have a 10-20 seconds of delay, in order to have all input files loaded.
```
ulimit -n 4096 # Make sure we can open sufficiently many files
~raw> readout.exe file:rdo_TF.cfg{code}
```
 
Upon data transfer to StfBuilder, readout will print the stats, like:
```
2020-06-23 18:07:59.003364 Last interval (1.00s): blocksRx=0, block rate=0.00, bytesRx=0, rate=0.000 b/s
2020-06-23 18:08:00.003382 Last interval (1.00s): blocksRx=2930, block rate=2930.00, bytesRx=1156508880, rate=9.252 Gb/s
2020-06-23 18:08:01.003384 Last interval (1.00s): blocksRx=0, block rate=0.00, bytesRx=0, rate=0.000 b/s{noformat}
```

StfBuilder will print one warning regarding the timeout on the last received TF. This can be ignored in this case. The log should look like :

``` 
{noformat}[2020-06-23 18:07:59.928][I] readout[0]: in: 1224 (1156.52 MB) out: 0 (0 MB)
[2020-06-23 18:08:01.733][W] READOUT INTERFACE: finishing STF on a timeout. stf_id=1 size=1156508880
[2020-06-23 18:08:02.607][I] Sending STF out. stf_id=1 channel=standalone-chan[0] stf_size=1156508880 unique_equipments=1224{noformat}
```

After this, both processes can be closed with Ctrl-C. The resulting TFs are stored in a new directory under TF_PATH (the name of the dir is the time of running)

 
