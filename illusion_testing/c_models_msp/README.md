# C models to run on the MSP430 on our Illusion System

These are the C source to run on the Illusion System. The modes of operation are combined in a single source, determined by a flag passed to the compiler. There are addtitional flags as well that were used for testing. The IOs from the golden models are used to compare and validate simulated execution.

Directory map:
- `mnist` : A simple CNN for MNIST image recognition. Total weights < 4KB
- `svhn` : A more complex CNN for SVHN image recognition. Total weights < 32KB
- `d2nn` : A dynamic CNN with two branches and built in decision node for MNIST image recognition. Total weights < 32KB
- `lstm` : A LSTM model for keyword spotting. Total weights < 32KB.

