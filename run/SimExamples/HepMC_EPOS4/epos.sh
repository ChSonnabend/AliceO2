#!/bin/bash
# Script based on CRMC example
# EPOS4 option files must contain ihepmc set to 2 to print HepMC
# data on stdout. -hepmc flag is not needed anymore, but -hepstd is fundamental
# in order not to print useless information on stdout (a z-*optns*.mtr file will be created)

optns="example"
seed=1
EPOS4=""

if [ -z "$EPO4VSN" ]; then
    # Error: EPO4VSN environment variable is not set
    exit 1
fi

if [ "$EPO4VSN" = "4.0.0" ]; then
    EPOS4="$EPOS4_ROOT/epos4/scripts/epos"
else
    EPOS4="$EPOS4_ROOT/bin/epos"
fi

while test $# -gt 0 ; do
    case $1 in
        -i|--input)   optns=$2 ; shift ;;
        -s|--seed)    seed=$2 ; shift ;;
        -h|--help) usage; exit 0 ;;
    esac
    shift
done

if [ ! -f $optns.optns ]; then
    echo "Error: Options file $optns.optns not found"
    exit 2
fi

if grep -Fq "set ihq 1" $optns.optns; then
    if [ -z "$EPO4HQVSN" ]; then
        # Error: EPOS4HQ version not found
        exit 3
    else
        # Running with EPOS4HQ
        EPOS4="$EPO4HQ/bin/eposhq"
    fi
fi

if [ $seed -eq 0 ]; then
    # Seed can't be 0, random number will be used
    seed="$RANDOM"
fi

# OR filters the stdout with only HepMC useful data
$EPOS4 -hepstd -s $seed $optns | sed -n 's/^\(HepMC::\|[EAUWVP] \)/\1/p'
