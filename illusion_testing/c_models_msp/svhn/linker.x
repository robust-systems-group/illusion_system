/* Default linker script, for normal executables */
OUTPUT_FORMAT("elf32-msp430","elf32-msp430","elf32-msp430")
OUTPUT_ARCH("msp430")
  /*data      (rwx)  	: ORIGIN = 0x0200, 	LENGTH = 4096*/
MEMORY
{
  instr      (rx)   	: ORIGIN = 0xd000,   LENGTH = 12288
  data_nvm  (rwx)  	: ORIGIN = 0x0200, 	LENGTH = 4096
  data      (rwx)  	: ORIGIN = 0x1200, 	LENGTH = 8192
  vectors64 (rw)   	: ORIGIN = 0xff80,      LENGTH = 0x40
  vectors32 (rw)   	: ORIGIN = 0xffc0,      LENGTH = 0x20
  vectors   (rw)   	: ORIGIN = 0xffe0,      LENGTH = 0x20
}
REGION_ALIAS("REGION_TEXT", instr);
REGION_ALIAS("REGION_DATA", data);
__WDTCTL = 0x0120;
__MPY    = 0x0130;
__MPYS   = 0x0132;
__MAC    = 0x0134;
__MACS   = 0x0136;
__OP2    = 0x0138;
__RESLO  = 0x013A;
__RESHI  = 0x013C;
__SUMEXT = 0x013E;

SECTIONS
{
  /* Read-only sections, merged into text segment.  */
  .hash          : { *(.hash)             }
  .dynsym        : { *(.dynsym)           }
  .dynstr        : { *(.dynstr)           }
  .gnu.version   : { *(.gnu.version)      }
  .gnu.version_d   : { *(.gnu.version_d)  }
  .gnu.version_r   : { *(.gnu.version_r)  }
  .rel.init      : { *(.rel.init) }
  .rela.init     : { *(.rela.init) }
  .rel.text      :
    {
      *(.rel.text)
      *(.rel.text.*)
      *(.rel.gnu.linkonce.t*)
    }
  .rela.text     :
    {
      *(.rela.text)
      *(.rela.text.*)
      *(.rela.gnu.linkonce.t*)
    }
  .rel.fini      : { *(.rel.fini) }
  .rela.fini     : { *(.rela.fini) }
  .rel.rodata    :
    {
      *(.rel.rodata)
      *(.rel.rodata.*)
      *(.rel.gnu.linkonce.r*)
    }
  .rela.rodata   :
    {
      *(.rela.rodata)
      *(.rela.rodata.*)
      *(.rela.gnu.linkonce.r*)
    }
  .rel.data      :
    {
      *(.rel.data)
      *(.rel.data.*)
      *(.rel.gnu.linkonce.d*)
    }
  .rela.data     :
    {
      *(.rela.data)
      *(.rela.data.*)
      *(.rela.gnu.linkonce.d*)
    }
  .rel.ctors     : { *(.rel.ctors)        }
  .rela.ctors    : { *(.rela.ctors)       }
  .rel.dtors     : { *(.rel.dtors)        }
  .rela.dtors    : { *(.rela.dtors)       }
  .rel.got       : { *(.rel.got)          }
  .rela.got      : { *(.rela.got)         }
  .rel.bss       : { *(.rel.bss)          }
  .rela.bss      : { *(.rela.bss)         }
  .rel.plt       : { *(.rel.plt)          }
  .rela.plt      : { *(.rela.plt)         }
  /* Internal text space.  */
  .text :
  {
    . = ALIGN(2);
    *(.init)
    *(.init0)  /* Start here after reset.  */
    *(.init1)
    *(.init2)  /* Copy data loop  */
    *(.init3)
    *(.init4)  /* Clear bss  */
    *(.init5)
    *(.init6)  /* C++ constructors.  */
    *(.init7)
    *(.init8)
    *(.init9)  /* Call main().  */
     __ctors_start = . ;
     *(.ctors)
     __ctors_end = . ;
     __dtors_start = . ;
     *(.dtors)
     __dtors_end = . ;
    . = ALIGN(2);
    *(.text)
    . = ALIGN(2);
    *(.text.*)
    . = ALIGN(2);
    *(.fini9)  /*   */
    *(.fini8)
    *(.fini7)
    *(.fini6)  /* C++ destructors.  */
    *(.fini5)
    *(.fini4)
    *(.fini3)
    *(.fini2)
    *(.fini1)
    *(.fini0)  /* Infinite loop after program termination.  */
    *(.fini)
     _etext = . ;
  }  > instr
  .rodata  :
  {
     PROVIDE (__rodata_start = .) ;
    . = ALIGN(2);
    *(.rodata)
    . = ALIGN(2);
    *(.gnu.linkonce.r*)
    . = ALIGN(2);
     _erodata = . ;
  } > data_nvm
  .bss  :
  {
     PROVIDE (__bssstart = .) ;
    *(.bss)
    *(COMMON)
     PROVIDE (__bssend = .) ;
  } > data_nvm
  .data  :
  {
     PROVIDE (__data_start = .) ;
    . = ALIGN(2);
    *(.data)
    . = ALIGN(2);
    *(.gnu.linkonce.d*)
    . = ALIGN(2);
     _edata = . ;
  }  > data_nvm
  .noinit  :
  {
     PROVIDE (__noinit_start = .) ;
    *(.noinit)
    *(COMMON)
     PROVIDE (__noinit_end = .) ;
     _end = . ;
  }  > data
  .vectors32  :
  {
     PROVIDE (__vectors32_start = .) ;
    *(.vectors32*)
     _vectors32_end = . ;
  }  > vectors32
  .vectors64  :
  {
     PROVIDE (__vectors64_start = .) ;
    *(.vectors64*)
     _vectors64_end = . ;
  }  > vectors64
  .vectors  :
  {
     PROVIDE (__vectors_start = .) ;
    *(.vectors*)
     _vectors_end = . ;
  }  > vectors
  /* Stabs debugging sections.  */
  .stab 0 : { *(.stab) }
  .stabstr 0 : { *(.stabstr) }
  .stab.excl 0 : { *(.stab.excl) }
  .stab.exclstr 0 : { *(.stab.exclstr) }
  .stab.index 0 : { *(.stab.index) }
  .stab.indexstr 0 : { *(.stab.indexstr) }
  .comment 0 : { *(.comment) }
  /* DWARF debug sections.
     Symbols in the DWARF debugging sections are relative to the beginning
     of the section so we begin them at 0.  */
  /* DWARF 1 */
  .debug          0 : { *(.debug) }
  .line           0 : { *(.line) }
  /* GNU DWARF 1 extensions */
  .debug_srcinfo  0 : { *(.debug_srcinfo) }
  .debug_sfnames  0 : { *(.debug_sfnames) }
  /* DWARF 1.1 and DWARF 2 */
  .debug_aranges  0 : { *(.debug_aranges) }
  .debug_pubnames 0 : { *(.debug_pubnames) }
  /* DWARF 2 */
  .debug_info     0 : { *(.debug_info) *(.gnu.linkonce.wi.*) }
  .debug_abbrev   0 : { *(.debug_abbrev) }
  .debug_line     0 : { *(.debug_line) }
  .debug_frame    0 : { *(.debug_frame) }
  .debug_str      0 : { *(.debug_str) }
  .debug_loc      0 : { *(.debug_loc) }
  .debug_macinfo  0 : { *(.debug_macinfo) }
  PROVIDE (__stack = ORIGIN(data) + LENGTH(data)) ;
  PROVIDE (__data_start_rom = _etext) ;
  PROVIDE (__data_end_rom   = _etext + SIZEOF (.rodata)) ;
  PROVIDE (__datastart = __rodata_start) ;
  PROVIDE (__data_start_ram = _erodata) ;
  PROVIDE (__bsssize = SIZEOF (.bss)) ;
  PROVIDE (__romdatastart  = ORIGIN(data_nvm)) ;
  PROVIDE (__romdatacopysize  = LENGTH(data_nvm)) ;
  PROVIDE (__data_end_ram   = _erodata + SIZEOF (.data)) ;
  PROVIDE (__noinit_start_rom = _etext + SIZEOF (.data)) ;
  PROVIDE (__noinit_end_rom = _etext + SIZEOF (.data) + SIZEOF (.noinit)) ;
}
