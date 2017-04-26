#ifndef __MEM_RUBY_STRUCTURES_SIMPLEPREFETCHER_HH__
#define __MEM_RUBY_STRUCTURES_SIMPLEPREFETCHER_HH__

#include "base/statistics.hh"
#include "mem/ruby/common/Address.hh"
#include "mem/ruby/slicc_interface/AbstractController.hh"
#include "mem/ruby/slicc_interface/RubyRequest.hh"
#include "params/SimplePrefetcher.hh"
#include "sim/sim_object.hh"

class SimplePrefetcher : public SimObject
{
    public:
        typedef SimplePrefetcherParams Params;
        SimplePrefetcher(const Params *p);
        ~SimplePrefetcher();
        
        void setController(AbstractController *_ctrl)
        { m_controller = _ctrl; }
    
        void observeMiss(Addr address, const RubyRequestType& type);
        void observePfHit(Addr address);
    
        void regStats();
    
        Stats::Scalar m_num_miss_observed;
        Stats::Scalar m_num_prefetch_late;
        Stats::Scalar m_num_prefetch_hit;
        Stats::Scalar m_num_prefetch_miss;
        Stats::Scalar m_num_prefetch_issued;
    
    private:
        AbstractController *m_controller;
        int m_pref_dist;
};

#endif // __MEM_RUBY_STRUCTURES_SIMPLEPREFETCHER_HH__
