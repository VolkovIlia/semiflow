/* QEMU `mps2-an385` (Cortex-M3) and `mps2-an386` (Cortex-M4F) memory map
 * (hw/arm/mps2.c): ZBT SSRAM1 4 MiB at 0x0000_0000 holds code and the vector
 * table; ZBT SSRAM2/3 4 MiB at 0x2000_0000 holds data, the heap and the stack.
 * Both are plain RAM under QEMU; the kernel image is loaded into SSRAM1. */
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 4M
  RAM   : ORIGIN = 0x20000000, LENGTH = 4M
}
