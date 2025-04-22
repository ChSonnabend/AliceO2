/// \author Marco Giacalone - March 2025

// A simple wrapper and demonstrator around Pythia8 for extracting HepMC3 files.

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"

using namespace o2::eventgen;

class HepMC3_Pythia8Wrapper : public GeneratorPythia8
{
 public:
  HepMC3_Pythia8Wrapper(std::string filename = "pythia8.hepmc") : GeneratorPythia8(), mFileName(filename)
  {
    // HepMC conversion object.
    mToHepMC = std::make_unique<Pythia8::Pythia8ToHepMC>();
    mToHepMC->setNewFile((filename == "" ? "pythia.hepmc" : filename));
  };
  ~HepMC3_Pythia8Wrapper() = default;

  bool importParticles() override
  {
    // events are written after the importParticles step
    // since some filtering is happening there
    auto ret = GeneratorPythia8::importParticles();
    if (ret) {
      LOG(info) << "Writing event to HepMC3 format";
      mToHepMC->writeNextEvent(mPythia);
    }
    return ret;
  };

 private:
  std::string mFileName = "pythia8.hepmc";
  std::unique_ptr<Pythia8::Pythia8ToHepMC> mToHepMC;
};

FairGenerator*
  hepmc_pythia8(std::string filename = "pythia8.hepmc")
{
  std::cout << "HepMC3_Pythia8Wrapper initialising with filename: " << filename << std::endl;
  auto py8 = new HepMC3_Pythia8Wrapper(filename);
  return py8;
}
