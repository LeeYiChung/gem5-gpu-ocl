from m5.SimObject import SimObject
from m5.params import *

class SimplePrefetcher(SimObject):
    type = 'SimplePrefetcher'
    cxx_class = 'SimplePrefetcher'
    cxx_header = "mem/ruby/structures/Prefetcher.hh"
    
    pref_dist = Param.Int32(1, "")
