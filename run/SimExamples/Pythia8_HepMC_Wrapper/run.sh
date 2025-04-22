#!/bin/bash

#
# Script doing Pythia8 event generation and writing these events into HepMC3 files
# (next to generating the usual MCTrack kinematics output).
#
# The script also performs a second event generation based on the generated HepMC3 files.
# In principle it should yield identical kinematics files.
#

NEVENTS=1000
SEED=11

o2-sim -j 1 -g external --configKeyValues 'GeneratorExternal.fileName=Pythia8HepMC3.macro;GeneratorExternal.funcName=hepmc_pythia8("skimmed.hepmc");GeneratorPythia8.config=${O2_ROOT}/share/Generators/egconfig/pythia8_inel.cfg' --seed ${SEED} --noGeant -o pythia8_skimmed -n ${NEVENTS}
o2-sim -j 1 -g external --configKeyValues 'GeneratorExternal.fileName=Pythia8HepMC3.macro;GeneratorExternal.funcName=hepmc_pythia8("unskimmed.hepmc");GeneratorPythia8.config=${O2_ROOT}/share/Generators/egconfig/pythia8_inel.cfg;GeneratorPythia8.includePartonEvent=true' --seed ${SEED} --noGeant -o pythia8_unskimmed -n ${NEVENTS}

# propagate generated hepmc file; it should produce the same kinematics as the original Pythia8
o2-sim -j 1 -g hepmc --configKeyValues="GeneratorFileOrCmd.fileNames=skimmed.hepmc" --vertexMode kNoVertex --noGeant -o fromhepmc_skimmed -n ${NEVENTS} --seed ${SEED}
o2-sim -j 1 -g hepmc --configKeyValues="GeneratorFileOrCmd.fileNames=unskimmed.hepmc" --vertexMode kNoVertex --noGeant -o fromhepmc_unskimmed -n ${NEVENTS} --seed ${SEED}
