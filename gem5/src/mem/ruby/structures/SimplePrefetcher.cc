#include "debug/SimplePrefetcher.hh"
#include "mem/ruby/structures/SimplePrefetcher.hh"

SimplePrefetcher*
SimplePrefetcherParams::create()
{
    return new SimplePrefetcher(this);
}

SimplePrefetcher::SimplePrefetcher(const Params *p) : SimObject(p), m_pref_dist(p->pref_dist)
{

}

SimplePrefetcher::~SimplePrefetcher()
{

}


void 
SimplePrefetcher::observeMiss(Addr address, const RubyRequestType& type) {
    
    Addr line_addr = makeLineAddress(address);
    DPRINTF(SimplePrefetcher, "Observed miss for %s\n", line_addr);
    if(m_pref_dist != 0) {
        Addr next_line_addr = makeNextStrideAddress(line_addr, m_pref_dist);
        DPRINTF(SimplePrefetcher, "Prefetch  for %s\n", next_line_addr);
        m_controller->enqueuePrefetch(next_line_addr, RubyRequestType_LD);
    }
}

void
SimplePrefetcher::observePfHit(Addr address)
{
    Addr line_addr = makeLineAddress(address);
    Addr next_line_addr = makeNextStrideAddress(line_addr, m_pref_dist);
    m_controller->enqueuePrefetch(next_line_addr, RubyRequestType_LD);
}

void
SimplePrefetcher::regStats()
{
    m_num_miss_observed
        .name(name() + ".miss_observed")
        .desc("number of misses observed")
        ;
    
    m_num_prefetch_late
        .name(name() + ".prefetch_late")
        .desc("number of prefetch late")
        ;
    
    m_num_prefetch_hit
        .name(name() + ".prefetch_hit")
        .desc("number of prefetch hit")
        ;
    
    m_num_prefetch_miss
        .name(name() + ".prefetch_miss")
        .desc("number of prefetch miss")
        ;
 
    m_num_prefetch_issued
        .name(name() + ".prefetch_issued")
        .desc("number of prefetch issued")
        ;
}
