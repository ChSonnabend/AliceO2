// Standalone regression fixture for TrackTransportTruth.h. No ROOT/O2 runtime needed.
#include "DetectorsBase/TrackTransportTruth.h"
#include <fstream>
#include <iostream>
using namespace o2::data::detail;
struct Track {
  int mother;
  uint32_t mask;
  float score;
  bool evaluated = true;
  bool getInhibited() const { return false; }
  bool getTransportModelShadow() const { return evaluated; }
  float getTransportModelScore() const { return score; }
  int getMotherTrackId() const { return mother; }
  uint32_t getHitMask() const { return mask; }
  int GetPdgCode() const { return 211; }
  int getProcess() const { return 0; }
  float Px() const { return 1; }
  float Py() const { return 0; }
  float Pz() const { return 1; }
  float Vx() const { return 0; }
  float Vy() const { return 0; }
  float Vz() const { return 0; }
  float T() const { return 0; }
  float getWeight() const { return 1; }
};
void check(bool value) { if (!value) throw std::runtime_error("Transport CSV regression failed"); }
int main(int argc, char** argv)
{
  if (argc != 2 && argc != 3) return 2;
  const bool rootsOnly = argc == 3;
  auto t = labelTransportTrees({-1, 2, 3, 0}, {0, 4, 0, 0}, 1);
  check(t[0].hitMask == 4 && t[0].descendants == 3 && t[1].depth == 3);
  t = labelTransportTrees({-1, 0, 1}, {0, 0, 2}, 2);
  check(t[0].hitMask == 0 && t[1].hitMask == 2 && t[2].simulationRoot == 1);
  for (auto mothers : {std::vector<int>{-1, 2, 1}, std::vector<int>{-1, 8}}) {
    bool failed = false;
    try { labelTransportTrees(mothers, std::vector<uint32_t>(mothers.size()), 1); }
    catch (const std::runtime_error&) { failed = true; }
    check(failed);
  }
  std::vector<int> deep(100000);
  for (int i = 0; i < static_cast<int>(deep.size()); ++i) deep[i] = i - 1;
  std::vector<uint32_t> masks(deep.size()); masks.back() = 8;
  t = labelTransportTrees(deep, masks, 1);
  check(t[0].hitMask == 8 && t[0].descendants == 99999 && t.back().depth == 99999);
  std::vector<Track> tracks{{-1,0,.1f},{0,0,.9f},{0,0,.8f},
                            {2,0,std::numeric_limits<float>::quiet_NaN()}, {0,0xffe00000u,.2f}, {4,0,-1.f,false}};
  t = labelTransportTrees({-1,0,0,2,0,4}, {0,0,0,0,1,0}, 2);
  check(t[0].descendants == 4 && t[2].hitlessRoot == 2 && t[3].hitlessRoot == 2 && t[5].hitlessRoot == 5);
  std::ofstream out(argv[1]);
  writeTransportCsvHeader(out);
  for (int event = 0; event < 8; ++event) writeTransportCsvEvent(out, tracks, 2, "model,\"test\"", "sgn", event+1, event, .8, rootsOnly);
  out.close();
  check(static_cast<bool>(out));
  std::cout << "Transport truth, deep trees, root boundaries and CSV fixture passed\n";
}
