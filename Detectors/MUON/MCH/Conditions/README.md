<!-- doxy
\page refDetectorsMUONMCHConditions Conditions
/doxy -->

# MCH Conditions

The condition data we have are :

- the DCS datapoints (for HV and LV)
- the Bad Channel list (obtained from pedestal calibration runs)
- the Reject list (manual input)

Those objects are stored at the following CCDB paths :

- MCH/Calib/HV
- MCH/Calib/LV
- MCH/Calib/BadChannel
- MCH/Calib/RejectList

# o2-mch-bad-channels-ccdb

The BadChannel and RejectList objects can be uploaded, e.g. for debug purposes, using the `o2-mch-bad-channels-ccdb` program:

```shell
$ o2-mch-bad-channels-ccdb --help
This program get/set MCH bad channels CCDB object
Usage:
  -h [ --help ]                         produce help message
  -c [ --ccdb ] arg (=http://localhost:6464)
                                        ccdb url
  --starttimestamp arg (=1677687518645) timestamp for query or put -
                                        (default=now)
  --endtimestamp arg (=1677773918645)   end of validity (for put) - default=1
                                        day from now
  -l [ --list ]                         list timestamps, within the given
                                        range, when the bad channels change
  -p [ --put ]                          upload bad channel object
  -r [ --referenceccdb ] arg (=http://alice-ccdb.cern.ch)
                                        reference ccdb url
  -u [ --upload-default-values ]        upload default values
  -t [ --type ] arg (=BadChannel)       type of bad channel (BadChannel or
                                        RejectList)
  -q [ --query ]                        dump bad channel object from CCDB
  -v [ --verbose ]                      verbose output
  -s [ --solar ] arg                    solar ids to reject
  -d [ --ds ] arg                       dual sampas indices to reject
  -e [ --de ] arg                       DE ids to reject
  -a [ --alias ] arg                    DCS alias (HV or LV) to reject
```

For instance, to create in a local CCDB a RejectList object which declares solar number 32 as bad, from Tuesday 1 November 2022 00:00:01 UTC to Saturday 31 December 2022 23:59:59, use:

```shell
$ o2-mch-bad-channels-ccdb -p -s 32 -t RejectList --starttimestamp 1667260801000 --endtimestamp 1672531199000
```

The program will search the reference CCDB (defined with `--referenceccdb`) for existing objects valid during this period and propose you to either overwrite them or update them. In the first case, a single object will be created, valid for the whole period, containing only the new bad channels. In the second case, as many objects as necessary will be created with appropriate validity ranges, adding the new bad channels to the existing ones.

# o2-mch-scan-hvlv-ccdb

the HV or LV DCS datapoints stored in the CCDB (http://alice-ccdb.cern.ch) can be scanned using the `o2-mch-scan-hvlv-ccdb` program:

```shell
$ o2-mch-scan-hvlv-ccdb -h
This program scans HV or LV channels looking for issues
Usage:
  -h [ --help ]                    produce help message
  -r [ --runs ] arg                run(s) to scan (comma separated list of runs
                                   or ASCII file with one run per line)
  -c [ --channels ] arg            channel(s) to scan ("HV" or "LV" or comma
                                   separated list of (part of) DCS aliases)
  --configKeyValues arg            Semicolon separated key=value strings to
                                   change HV thresholds
  -d [ --duration ] arg (=0)       minimum duration (ms) of HV/LV issues to
                                   consider
  -i [ --interval ] arg (=30)      creation time interval (minutes) between
                                   CCDB files
  -w [ --warning ] arg (=1)        warning level (0, 1 or 2)
  -p [ --print ] arg (=1)          print level (0, 1, 2 or 3)
  -o [ --output ] arg (=scan.root) output root file name
```

It takes as input a list of runs and a list of either HV or LV channels to scan. **Note that it will scan the CCDB from the begining of the first run to the end of the last one, which can represent quite of lot of files.** More details about the options are given below.

It produces as output a list of detected issues, with time, duration and affected runs, and a root file with the displays of the data points per channel per chamber for a visual inspection. Issues are triggered when HV/LV values go below a given threshold. For HV channels it also compares the issues found by the internal algorithm with the ones found by [Detectors/MUON/MCH/Status/src/HVStatusCreator.cxx](../Status/src/HVStatusCreator.cxx).

For instance, to scan all HV channels for runs 545222 and 545223 and detect issues of a minimum duration of 10s, use:
```shell
o2-mch-scan-hvlv-ccdb -r 545222,545223 -c HV -d 10000
```

### channel input formats:
* "HV" to scan all HV channels
* "LV" to scan all LV channels
* comma separated list of (part of) DCS aliases, which must be all of the same type, i.e contain either Quad/Slat (type = HV), or Group/an/di/Sol (type = LV)

### warning levels:
* 0: no warning
* 1: check data points timestamp w.r.t. HV/LV file validity range with ±5s tolerance
* 2: check data points timestamp w.r.t. HV/LV file validity range without tolerance

### print levels:
* 0: print detected issues
* 1: same as 0 + print validity range of runs and HV/LV files
* 2: same as 1 + print the first and last data points of each selected channel
* 3: same as 1 + print all the data points of each selected channel
