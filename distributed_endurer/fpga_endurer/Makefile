CC = gcc
CCFLAGS = -g -lm -Wall -std=c11
LDFLAGS = -lpthread
TESTS = test_single_chip test_two_chips test_gemm_dist test_distributed test_endurer_top
MODELS = simulate_workload_svhn simulate_workload_d2nn simulate_workload_lstm2

.PHONY: all

all: $(TESTS) $(MODELS)

test: $(TESTS)

models: $(MODELS)

endurer.o: endurer.c endurer.h
	$(CC) $(CCFLAGS) -c endurer.c -o endurer.o

test_gemm_dist: test_gemm_dist.c mock_board.c endurer.c endurer.h mock_board.h
	$(CC) $(CCFLAGS) endurer.c mock_board.c test_gemm_dist.c -o test_gemm_dist
	./test_gemm_dist

test_distributed: test_distributed.c mock_board.c endurer.c endurer.h mock_board.h
	$(CC) $(CCFLAGS) endurer.c mock_board.c test_distributed.c -o test_distributed
	./test_distributed

simulate_workload_svhn: simulate_workload_svhn.c mock_board.c endurer.c endurer.h mock_board.h
	$(CC) $(CCFLAGS) endurer.c mock_board.c simulate_workload_svhn.c -o simulate_workload_svhn
	./simulate_workload_svhn

simulate_workload_d2nn: simulate_workload_d2nn.c mock_board.c endurer.c endurer.h mock_board.h
	$(CC) $(CCFLAGS) endurer.c mock_board.c simulate_workload_d2nn.c -o simulate_workload_d2nn
	./simulate_workload_d2nn

simulate_workload_lstm2: simulate_workload_lstm2.c mock_board.c endurer.c endurer.h mock_board.h
	$(CC) $(CCFLAGS) endurer.c mock_board.c simulate_workload_lstm2.c -o simulate_workload_lstm2
	./simulate_workload_lstm2

test_endurer_top: test_endurer_top.c endurer_top.c mock_board.c endurer.c endurer_top.h endurer.h mock_board.h
	$(CC) $(CCFLAGS) test_endurer_top.c endurer_top.c endurer.c mock_board.c -o test_endurer_top 
	./test_endurer_top

test_single_chip: test_single_chip.c mock_board.c endurer.c endurer.h mock_board.h
	$(CC) $(CCFLAGS) endurer.c mock_board.c test_single_chip.c -o test_single_chip
	./test_single_chip

test_two_chips: test_two_chips.c mock_board.c endurer.c endurer.h mock_board.h
	$(CC) $(CCFLAGS) endurer.c mock_board.c test_two_chips.c -o test_two_chips
	./test_two_chips

clean:
	rm -f $(TESTS) $(MODELS) *.o
