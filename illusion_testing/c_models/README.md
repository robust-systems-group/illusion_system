# C Gold Models

These are the C code golden models for testing the DNNs natively. Taking the output of the Illusion mapping, the workloads are separated in code accordingly.

Each model is in a separate directory.

Directory map:
- `mnist` : A simple CNN for MNIST image recognition. Total weights < 4KB
- `svhn` : A more complex CNN for SVHN image recognition. Total weights < 32KB
- `d2nn` : A dynamic CNN with two branches and built in decision node for MNIST image recognition. Total weights < 32KB
- `lstm` : A LSTM model for keyword spotting. Total weights < 32KB.

