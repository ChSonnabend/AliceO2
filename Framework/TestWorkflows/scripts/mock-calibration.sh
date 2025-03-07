#/bin/sh -ex
export DPL_SIGNPOSTS="calibration"
stage/bin/o2-dpl-raw-proxy --exit-transition-timeout 20 --data-processing-timeout 10 --dataspec "tst:TST/A/0" --channel-config "readout-proxy:address=tcp://0.0.0.0:4200,method=connect,type=pair" | \
  stage/bin/o2-testworkflows-simple-processor --exit-transition-timeout 20 --data-processing-timeout 10 --name reconstruction --processing-delay 5000 --eos-dataspec tst3:TST/C/0 --in-dataspec "tst2:TST/A/0" --out-dataspec "tst:TST/B/0" | \
  stage/bin/o2-testworkflows-simple-processor --exit-transition-timeout 20 --data-processing-timeout 10 --name calibration --processing-delay 1000 --in-dataspec "tst2:TST/C/0?lifetime=sporadic" --out-dataspec "tst:TCL/C/0?lifetime=sporadic" | \
  stage/bin/o2-testworkflows-simple-sink --exit-transition-timeout 20 --data-processing-timeout 10 --name calibration-publisher --dataspec "tst2:TCL/C/0?lifetime=sporadic" | \
  stage/bin/o2-testworkflows-simple-sink --exit-transition-timeout 20 --data-processing-timeout 10 --dataspec "tst:TST/B/0"
