end_of_test:
	nop
	br #0xffff

.section .vectors, "a"
.word end_of_test  ; Interrupt  0 (lowest priority)    <unused>
.word end_of_test  ; Interrupt  1                      <unused>
.word end_of_test  ; Interrupt  2                      <unused>
.word end_of_test  ; Interrupt  3                      <unused>
.word end_of_test  ; Interrupt  4                      <unused>
.word end_of_test  ; Interrupt  5                      <unused>
.word end_of_test  ; Interrupt  6                      <unused>
.word end_of_test  ; Interrupt  7                      <unused>
.word end_of_test  ; Interrupt  8                      <unused>
.word end_of_test  ; Interrupt  9                      <unused>
.word end_of_test  ; Interrupt 10                      Watchdog timer
.word end_of_test  ; Interrupt 11                      <unused>
.word end_of_test  ; Interrupt 12                      <unused>
.word end_of_test  ; Interrupt 13                      <unused>
.word end_of_test  ; Interrupt 14                      NMI
.word boot         ; Interrupt 15 (highest priority)   RESET
