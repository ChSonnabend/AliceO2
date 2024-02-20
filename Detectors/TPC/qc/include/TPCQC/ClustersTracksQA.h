#define AliceO2_TPC_QC_CLUSTERSTRACKSQA_H

#include "DataFormatsTPC/Defs.h"

namespace o2
{
namespace tpc
{

class TrackTPC;
class ClusterNativeAccess;

namespace qc
{
  
class ClustersTracksQa
{
    public:
    
        ClustersTracksQa() = default;

        template<class T>
        void ReadClustersFromRootFile(std::string fileName, std::vector<T>& clusters);

        ClusterNativeAccess loadClusters(std::string_view fileName = "tpc-native-clusters.root");
        std::vector<TrackTPC> loadTracks(std::string_view inputFileName = "tpctracks.root");

        GlobalPosition2D convertSecRowPadToXY(int sector, int row, float pad);

    private:

        ClassDefNV(ClustersTracksQa, 1)
};

}
}
}
