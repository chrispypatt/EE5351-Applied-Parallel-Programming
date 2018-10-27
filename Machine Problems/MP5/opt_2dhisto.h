#ifndef OPT_KERNEL
#define OPT_KERNEL

#define BLOCK_SIZE 1024

void opt_2dhisto(size_t height, size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH]);

/* Include below the function headers of any other functions that you implement */
void CopyBinsFromDeviceArray(uint8_t Ahost[HISTO_HEIGHT*HISTO_WIDTH], size_t height, size_t width,uint8_t* Adevice);
void CopyInputToDeviceArray(uint32_t* Adevice, size_t width, uint32_t* Ahost);

void initData(uint32_t *input[], size_t height, size_t width);
void destructData();

#endif
