# Distributed Endurer Data `
Directory map:
- `raw` : Traces read off of the RRAM chip at different time snapshots. Chip attempts to set all bits to 1 and
saves the actual bits read. Then chip attempts to reset all bits to 0 and saves actual bits read.
- `processed` : Cleaned data -> remove temporary write failures,
since they are not endurance related. Saves as same format of a trace of set bits and reset bits.
- `extracted_accuracy` : Scripts take the set and reset traces as masks on the model and
activation parts of memory, applied at different random offsets to model accuracy at different time
points in the snapshot, and evaluate accuracy on the masked model.

## Recreation steps
To go from the `raw` data and create `processed` data:
```
cd ../endurer_tests
bash clean_cycles.sh
```

To go from `processed` data and generate accuracy extraction results:
```
cd extracted_accuracy
bash run_all.sh
```

This script will take a while to extract the model from the time snapshot and evaluate accuracy on each model's test set. This is done for both Distributed Endurer meeasurements and No Resiliency (NR) measurements.

## Failure types
- Temporary write failures: any errors in the trace which resolve themselves later (anytime before `tenyear`)
are cleaned out at all previous time steps
- Endurance failures: any errors which do not resolve themselves later are considered stuck in time and *not*
cleaned out. The number of stuck bits accumulates over time, and such bits are used to create bit masks on the models in the
`extracted_accuracy` folder.

