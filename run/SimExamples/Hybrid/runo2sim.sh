#!/usr/bin/env bash
#
# Hybrid generator simulation example:
# the simulation is configured using a JSON file (hybridconfig.json in this folder), whose
# template can be generated using the script ${O2DPG_ROOT}/MC/bin/o2_hybrid_gen.py
set -x
if [ ! "${O2DPG_ROOT}" ]; then
    echo "This needs O2DPG loaded; alienv enter ..."
    exit 1
fi

[ ! "${O2_ROOT}" ] && echo "Error: This needs O2 loaded" && exit 2

NEV=-1
more=""
JOBS=2

usage()
{
    cat <<EOF
Usage: $0 [OPTIONS]

Options:

  -m,--more    CONFIG      More configurations ($more)
  -n,--nevents EVENTS      Number of events ($NEV)
  -j,--jobs    JOBS        Number of jobs ($JOBS)
  -h,--help                Print these instructions
  --                       Rest of command line sent to o2-sim

COMMAND must be quoted if it contains spaces or other special
characters

Below follows the help output of o2-sim

EOF
}

if [ "$#" -lt 2 ]; then
    echo "Running with default values"
fi

while test $# -gt 0 ; do
    case $1 in
        -m|--more)    more="$2" ; shift ;;
        -n|--nevents) NEV=$2 ; shift ;;
        -j|--jobs)    JOBS=$2 ; shift ;;
        -h|--help) usage; o2-sim --help full ; exit 0 ;;
        --)           shift ; break ;;
        *) echo "Unknown option '$1', did you forget '--'?" >/dev/stderr
           exit 3
           ;;
    esac
    shift
done

# Set number of events in optns file
if [ ! $NEV -eq -1 ]; then
    echo "Setting number of events to $NEV"
else
    echo "Number of events not set, defaulting to 10..."
    NEV=10
fi

# Generation of 1000 events using STARlight in a slight.hepmc file
${O2_ROOT}/examples/HepMC_STARlight/run-starlight.sh

# Generation of event pool with pythia8 (10000 events) in a evtpool.root file
${O2DPG_ROOT}/MC/run/examples/event_pool.sh --make

# Starting simulation with Hybrid generator
${O2_ROOT}/bin/o2-sim --noGeant -j $JOBS --field ccdb --vertexMode kCCDB --run 300000 --configKeyValues "MFTBase.buildAlignment=true;GeneratorHybrid.configFile=$PWD/hybridconfig.json;GeneratorHybrid.randomize=false;${more}" -g hybrid -o genevents --timestamp 1546300800000 --seed 836302859 -n $NEV