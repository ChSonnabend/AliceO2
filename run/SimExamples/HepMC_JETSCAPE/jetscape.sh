#!/bin/sh
# Script based on EPOS4 example
# This script is used to run JETSCAPE with the given XML file
# setting the seed and HepMC output filename. Contrary to the
# epos example, the HepMC output is generated in a custom named file
# not passing from the stdout.

xml="example"
seed=$RANDOM
hepmc="jetout.hepmc"

usage()
{
    cat <<EOF
Usage: $0 [OPTIONS]

Options:

  -i,--input   INPUT       XML user file fed to JETSCAPE ($xml)
  -o,--output  OUTPUT      HepMC output file ($hepmc)
  -s,--seed    SEED        RNG seed ($seed)
  -h,--help                Print these instructions
  --                       Rest of command line sent to o2-sim

EOF
}

while test $# -gt 0 ; do
    case $1 in
        -i|--input)   xml=$2 ; shift ;;
        -o|--output)  hepmc=$2 ; shift ;;
        -s|--seed)    seed=$2 ; shift ;;
        -h|--help) usage; exit 0 ;;
    esac
    shift
done

if [ ! -f $xml.xml ]; then
    echo "Error: Options file $xml.xml not found"
    exit 1
fi

if [ $seed -eq 0 ]; then
    echo "Seed can't be 0, random number will be used"
    seed=$RANDOM
else
    if grep -Fq "<seed>" $xml.xml; then
        sed -i "/<seed>/c\  <seed>$seed</seed>" $xml.xml
    else
        sed -i "/<\/jetscape>/i\  <Random>\n    <seed>$seed</seed>\n  </Random>" $xml.xml
    fi
    echo "Seed set to $seed"
fi

# Check if hepmc output has been set
if [ ! -z "$hepmc" ]; then
    # Remove extension
    newhep=$(echo $hepmc | sed 's/.hepmc//')
    if grep -Fq "<outputFilename>" $xml.xml; then
        sed -i "/<outputFilename>/c\  <outputFilename>$newhep</outputFilename>" $xml.xml
    else
        sed -i "/<jetscape>/a\  <outputFilename>$newhep</outputFilename>" $xml.xml
    fi
    echo "HepMC output file set to $hepmc"
else
    echo "Error: HepMC output file not set"
    exit 2
fi

# Master XML file pulled directly from the JETSCAPE directory
runJetscape $xml.xml $JETSCAPE_ROOT/config/jetscape_master.xml