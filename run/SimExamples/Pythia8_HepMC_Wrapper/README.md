<!-- doxy
\page refrunSimExamplesPythiaHepMCWrapper Example showing easy HepMC extraction using GeneratorPythia8
/doxy -->

This example demonstrates how we can extend GeneratorPythia8 in a user-defined macro (or external generator),
to achieve additional HepMC3 export of generated Pythia8 events.

The example provides a small utility for poeple in need to obtain HepMC files from Pythia8.
Note that many other methods to achieve this are possible (See original Pythia8 example).

The example provides:

- The external generator implementation `Pythia8HepMC3.C`
- a `run.sh` script demonstrating it's usage and a check feeding back the generated hepmc into the simulation


