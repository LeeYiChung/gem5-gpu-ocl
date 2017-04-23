#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x0000000000000346,0x0000004801000002,0x0000000000000278\n"
".quad 0x0000000000000000,0x0000001400010004,0x0000000900000038,0x0000000000000015\n"
".quad 0x0000000000000000,0x632e656369766564,0x0000000000000075,0x33010102464c457f\n"
".quad 0x0000000000000004,0x0000000100be0002,0x0000000000000000,0x0000000000000208\n"
".quad 0x0000000000000040,0x0038004000140114,0x0001000400400002,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000300000001\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000140,0x0000000000000036\n"
".quad 0x0000000000000000,0x0000000000000004,0x0000000000000000,0x000000030000000b\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000176,0x0000000000000001\n"
".quad 0x0000000000000000,0x0000000000000001,0x0000000000000000,0x0000000200000013\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000177,0x0000000000000090\n"
".quad 0x0000000600000002,0x0000000000000001,0x0000000000000018,0x7472747368732e00\n"
".quad 0x747274732e006261,0x746d79732e006261,0x672e766e2e006261,0x6e692e6c61626f6c\n"
".quad 0x672e766e2e007469,0x0000006c61626f6c,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000010003000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000020003000000,0x0000000000000000,0x0000000000000000,0x0000030003000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000003000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000003000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000500000006,0x0000000000000208,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000070,0x0000000000000070,0x0000000000000004,0x0000000560000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000207\n"
".quad 0x0000000000000207,0x0000000000000004,0x0000005001000001,0x0000000000000036\n"
".quad 0x0000003800000000,0x0000001400030000,0x0000000900000040,0x0000000000000015\n"
".quad 0x0000000000000000,0x0000000000000000,0x632e656369766564,0x0000000000000075\n"
".quad 0x762e0a0a0a0a0a0a,0x33206e6f69737265,0x677261742e0a302e,0x30325f6d73207465\n"
".quad 0x7365726464612e0a,0x3620657a69735f73,0x0000000a0a0a0a34\n"
".text");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[107];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const struct {int m; int v; const unsigned long long* d; char* f;} __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 1, fatbinData, 0 };
#ifdef __cplusplus
}
#endif