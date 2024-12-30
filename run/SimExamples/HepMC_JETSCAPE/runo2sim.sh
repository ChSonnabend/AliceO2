#!/usr/bin/env bash
#
# This is a simple simulation example showing how to
# start JETSCAPE generation automatically using cmd with hepmc output on FIFO
# and simultaneosly use o2-sim for transport

# JETSCAPE and O2 must be loaded
set -x
if [ ! "${JETSCAPE_ROOT}" ]; then
    echo "This needs JETSCAPE loaded; alienv enter ..."
    exit 1
fi

[ ! "${O2_ROOT}" ] && echo "Error: This needs O2 loaded" && exit 2

cmd="$PWD/jetscape.sh"
NEV=-1
more=""
xml="example"
JOBS=2

usage()
{
    cat <<EOF
Usage: $0 [OPTIONS]

Options:

  -m,--more    CONFIG      More configurations ($more)
  -n,--nevents EVENTS      Number of events ($nev)
  -i,--input   INPUT       XML configuration file fed to JETSCAPE ($xml)
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
        -i|--input)   xml=$2 ; shift ;;
        -j|--jobs)    JOBS=$2 ; shift ;;
        -h|--help) usage; o2-sim --help full ; exit 0 ;;
        --)           shift ; break ;;
        *) echo "Unknown option '$1', did you forget '--'?" >/dev/stderr
           exit 3
           ;;
    esac
    shift
done

echo "XML User file: $xml"

if [ ! -f $xml.xml ]; then
    echo "Error: Options file $xml.xml not found"
    exit 4
fi

# Set number of events in the XML file
if [ ! $NEV -eq -1 ]; then
    echo "Setting number of events to $NEV"
    if grep -Fq "<nEvents>" $xml.xml; then
        sed -i "/<nEvents>/c\  <nEvents>$NEV</nEvents>" $xml.xml
    else
        sed -i "/<jetscape>/a\  <nEvents>$NEV</nEvents>" $xml.xml
    fi
else
    echo "Number of events not set, checking xml file..."
    if grep -Fq "<nEvents>" $xml.xml; then
        NEV=$(grep -F "<nEvents>" $xml.xml | awk '{print $2}')
        echo "Number of events set to $NEV"
    else
        echo "Error: Number of events not set in JETSCAPE"
        exit 5
    fi
fi

# Starting simulation
o2-sim -j $JOBS -n ${NEV} -g hepmc --seed $RANDOM  \
       --configKeyValues "GeneratorFileOrCmd.cmd=$cmd -i $xml;GeneratorFileOrCmd.fileNames=test_out.hepmc;GeneratorFileOrCmd.outputSwitch=-o;GeneratorFileOrCmd.bMaxSwitch=none;GeneratorFileOrCmd.nEventsSwitch=none;${more}"
