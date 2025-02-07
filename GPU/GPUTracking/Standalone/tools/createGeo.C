#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TSystem.h>
#include "TRDBase/GeometryFlat.h"
#include "DetectorsBase/GeometryManager.h"
#include "TRDBase/Geometry.h"
#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainTracking.h"

using namespace o2::gpu;

void createGeo()
{
  o2::base::GeometryManager::loadGeometry();
  auto gm = o2::trd::Geometry::instance();
  gm->createPadPlaneArray();
  gm->createClusterMatrixArray();
  o2::trd::GeometryFlat gf(*gm);
  gSystem->Load("libO2GPUTracking");
  GPUReconstruction* rec = GPUReconstruction::CreateInstance(GPUReconstruction::DeviceType::CPU);
  GPUChainTracking* chain = rec->AddChain<GPUChainTracking>();
  chain->SetTRDGeometry(&gf);
  rec->DumpSettings();
}

#endif
