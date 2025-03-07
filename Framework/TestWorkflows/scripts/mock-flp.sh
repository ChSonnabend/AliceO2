#/bin/sh -ex
stage/bin/o2-testworkflows-simple-source --dataspec tst:TST/A/0 --delay 1000 | \
  stage/bin/o2-dpl-output-proxy --dataspec "tst:TST/A/0" --channel-config "downstream:address=tcp://0.0.0.0:4200,method=bind,type=pair"
