<!-- doxy
\page refrunSimExamplesHybrid Example Hybrid
/doxy -->

The usage of the Hybrid generator with the o2-sim is presented in this short manual.
All the other generators are implemented as sub-generators and they can be called thanks to a
JSON file, fed to o2-sim via the GeneratorHybrid.configFile parameter. The O2sim package needs to be loaded in order to use this example.

The example can be run automatically using the runo2sim.sh script, which contains most of the
available generators in O2. The JSON template can be generated using the ${O2DPG_ROOT}/MC/bin/o2_hybrid_gen.py script. To use this example the user can simply copy the entire Hybrid example folder and execute the script after giving it execution permissions (`chmod +x runo2sim.sh`).

# Files description

- **runo2sim.sh** &rarr; allows to use the hybrid generator example
- **hybridconfig.json** &rarr; example JSON file for the hybrid generator configuration
- **example.optns** &rarr; options file to be used in EPOS4 implemented as subgenerator in this example (the .optns must be available in the current working directory)
- **evtpool.root** &rarr; cached events to be used with the extkinO2 generator
- **epos4.hepmc** &rarr; EPOS4 events stored as hepmc file