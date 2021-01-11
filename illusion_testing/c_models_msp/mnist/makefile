# makefile configuration
NAME            = mnist
OBJECTS         = main.obj test.obj tensor.obj omsp_func.obj
#OBJECTS         = main.obj omsp_func.obj

WARNINGS        =  
ARCH =  -mmcu=msp430
#CFLAGS         = -D ${CHIP_ID} -O2 ${WARNINGS}  ${ARCH}   -mhwmult=16bit    # Old flags
CFLAGS         = -D ${CHIP_ID} -D ${MODE} -D${CODETIME} -O2 ${WARNINGS}  ${ARCH}   -mhwmult=16bit    # Old flags
LDFLAGS = -L/scratch0/tonyfwu/LETI_Tapeout/openmsp430/gcc/usr/local/lib -L/scratch0/tonyfwu/LETI_Tapeout/openmsp430/gcc/usr/local/msp430-elf/lib -L/scratch0/tonyfwu/LETI_Tapeout/openmsp430/gcc/usr/local/lib/gcc/msp430-elf/5.3.0  -lmul_16 -lgcc -lc
#switch the compiler (for the internal make rules)

CC              = msp430-elf-gcc-5.3.0
LD              = msp430-ld
AS              = msp430-as
DUMP              = msp430-objdump


.PHONY: all FORCE clean 

#all should be the first target. it's built when make is runwithout args
all: ${NAME}.elf ${NAME}.a43 ${NAME}.lst

#additional rules for files

%.obj : %.s
	${AS} -alsm $^ -o $@ > asm.log

%.s : %.c
	${CC} ${CFLAGS} -S $^

${NAME}.elf: ${OBJECTS}
	${LD} -T linker.x -o $@ $^ ${LDFLAGS}

${NAME}.a43: ${NAME}.elf
	msp430-objcopy -O ihex $^ $@

${NAME}.lst: ${NAME}.elf
	msp430-objdump -dSt $^ >$@

clean:
	rm -f ${NAME} ${NAME}.a43 ${NAME}.lst ${NAME}.elf *.o *.s *.obj

#dummy target as dependecy if something has to be build everytime
FORCE:


