#include "DetectorsBase/TrackTransportTruth.h"
#include "DetectorsBase/TrackTransportProtection.h"
#include <sstream>
using namespace o2::data::detail;
void check(bool x) { if (!x) throw std::runtime_error("Contribution regression failed"); }
struct Track {
 int mother; uint32_t contribution; int protection; int version=1;
 bool getInhibited() const { return false; }
 bool getTransportModelShadow() const { return true; }
 float getTransportModelScore() const { return .99f; }
 int getMotherTrackId() const { return mother; }
 uint32_t getHitMask() const { return 0; } // This signal is invisible to the old target.
 uint32_t getDetectorContributionMask() const { return contribution; }
 int getDetectorContributionVersion() const { return version; }
 int getTransportModelProtection() const { return protection; }
 int GetPdgCode() const { return 50000050; }
 int getProcess() const { return 0; }
 float Px() const { return 1; } float Py() const { return 0; } float Pz() const { return 1; }
 float Vx() const { return 0; } float Vy() const { return 0; } float Vz() const { return 0; }
 float T() const { return 0; } float getWeight() const { return 1; }
};
int main() {
 check(trackTransportProtection(50000050,true,true,true)==1);
 check(trackTransportProtection(50000051,false,true,true)==3);
 check(trackTransportProtection(11,false,true,true)==2);
 check(trackTransportProtection(22,true,true,true)==0);
 check(trackTransportProtection(11,false,true,false)==0);
 check(trackTransportProtection(50000050,false,false,false)==0);
 auto tree=labelTransportTrees({-1,0,1},{0,0,4},1);
 check(tree[0].hitMask==4 && tree[1].hitMask==4 && tree[2].hitMask==4);
 std::vector<Track> tracks{{-1,0,0},{0,4,3},{0,0,3}};
 std::ostringstream out;writeTransportCsvHeader(out);writeTransportCsvEvent(out,tracks,1,"test","sgn",1,0,.9);
 check(out.str().find("own_contribution_mask")!=std::string::npos);
 check(out.str().find("effective_would_reject")!=std::string::npos);
 check(out.str().find(",1,0,3,1,0\n")!=std::string::npos); // protected rejection is suppressed.
 std::ostringstream noCut;writeTransportCsvEvent(noCut,tracks,1,"test","sgn",1,0,-1.);
 check(noCut.str().find(",1,0,3,-1,-1\n")!=std::string::npos);
 tracks[1].version=0;bool refused=false;
 try {writeTransportCsvEvent(out,tracks,1,"test","sgn",1,0,.9);} catch(const std::runtime_error&) {refused=true;}
 check(refused);
}
