<!-- doxy
\page refrunSimExamplesHepMC_JETSCAPE Example HepMC_JETSCAPE
/doxy -->

The usage of JETSCAPE with the O2 machinery is presented in this short manual.
An in-depth explanation of the mechanisms behind the HepMC(3) data handling can be found in the
HepMC_fifo folder of the MC examples. The scripts use the `cmd` parameter of `GeneratorHepMC`
to spawn the JETSCAPE generation via the `jetscape.sh` script. It is important to turn on the
HepMC3 output format in the xml configuration file, as done in jetscape_user_example.xml, otherwise
the simulation will not work.

# Scripts description

Two scripts are available to run the simulations
- **jetscape.sh** &rarr; starts the actual JETSCAPE generation
- **runo2sim.sh** &rarr; allows the generation of events using o2-sim

In addition an jetscape_user_example.xml file is provided to start JETSCAPE with user parameters.
The user could easily create scripts similar to the one provided for the EPOS4 tutorial for DPL or O2DPG
based simulations.

## jetscape.sh

It can be run without the help of the other scripts to simply generate an .hepmc file.
This example shows all the functionalities of the script (which are implemented in a similar way inside
the generation steering scripts). In particular the `-i` flag allows to provide the .xml user configuration file to JETSCAPE, `-s` feeds the generator with a user seed, and the HepMC output filename is set using the `-o` flag. The script edits automatically some specific parts of the provided input XML file.

## runo2sim.sh

This script works only with O2sim versions containing the FIFO custom name creation fix (the specific build will be added here in the future) otherwise it will crash or not complete the simulation.
Few flags are available to change the settings of the generation:
- **-m , --more** &rarr; feeds the simulation with advanced parameters provided to the configuration key flags
- **-n , --nevents** &rarr; changes the number of events in the .xml file or gets the one in the file if no events are provided
- **-i , --input** &rarr; .xml filename to feed JETSCAPE, no extension must be set in the filename
- **-j , --jobs** &rarr; sets the number of workers (jobs)
- **-h , --help** &rarr; prints usage instructions

The last few lines of the script contain the execution of o2-sim, so this part can be modified by the users following their requirements. It's important not to delete from the configuration keys `GeneratorFileOrCmd.cmd=$cmd -i $xml;GeneratorFileOrCmd.fileNames=test_out.hepmc;GeneratorFileOrCmd.outputSwitch=-o;GeneratorFileOrCmd.bMaxSwitch=none;GeneratorFileOrCmd.nEventsSwitch=none;` because the script might not work anymore, and it would be better to provide additional configurations via the -m flag.

