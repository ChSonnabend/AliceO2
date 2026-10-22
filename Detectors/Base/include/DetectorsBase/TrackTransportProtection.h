#ifndef O2_TRACK_TRANSPORT_PROTECTION_H
#define O2_TRACK_TRANSPORT_PROTECTION_H
#include <cstdint>
namespace o2::data::detail
{
// Bit 0: optical photon; bit 1: transport secondary. Neither needs future information.
inline uint8_t trackTransportProtection(int pdg, bool isInputRoot, bool protectOptical, bool protectSecondaries)
{
  const bool optical = pdg == 50000050 || pdg == -50000050 || pdg == 50000051 || pdg == -50000051;
  return (protectOptical && optical ? 1 : 0) | (protectSecondaries && !isInputRoot ? 2 : 0);
}

}
#endif
