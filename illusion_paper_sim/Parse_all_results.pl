#!/usr/bin/perl
#
# Copyright (C) 2020 by The Board of Trustees of Stanford University
# This program is free software: you can redistribute it and/or modify it under
# the terms of the Modified BSD-3 License as published by the Open Source
# Initiative.
# If you use this program in your research, we request that you reference the
# Illusion paper, and that you send us a citation of your work.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the BSD-3 License for more details.
# You should have received a copy of the Modified BSD-3 License along with this
# program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
#


use strict;
use warnings;

# Note -- $ARGV[0] is the first argument to the script, *not* the script itself
# (*not* like C's argv[0])
my $network_name=$ARGV[0];
my $param_batch_size=$ARGV[1];
my $baseline_config=$ARGV[2];
my $comparison_config=$ARGV[3];
my $NN_DATAFLOW_PATH=$ENV{'NN_DATAFLOW_PATH'};
my $op_directory="ops";
my $results="results";
system("mkdir -p ./$op_directory");
system("mkdir -p ./$results");
my @word_size=(16);
my @batch_size=($param_batch_size);
my @config_range=("Baseline", $comparison_config); # Always compare vs. Baseline
foreach my $word (@word_size){
	foreach my $batch (@batch_size){
		my $schedule_name=$network_name."_".$word."_".$batch;
        #my $Operations_file_name=$network_name."_".$word."_".$batch."_Ops.pl";
        my $Operations_file_name=$comparison_config."_".$network_name."_ops.pl";
		my $multipliers_file_name=$network_name."_multipliers.pl";
#		my $iter=0;
#		foreach my $config (@config_range){	
#			my $schedule_directory="$NN_DATAFLOW_PATH/mod_schedule/$config";
#			system("./Extract_operations.pl $schedule_directory/$schedule_name ./$op_directory/$Operations_file_name $network_type $iter");
#			$iter=$iter+1;
#                }
        # TODO unify $op_file_name with $comparison_config as we pass to Acc_results_compare_trace.pl
		my $output_file=$network_name."_".$word."_".$batch."_".$comparison_config.".csv";
        #print "./Acc_results_compare_trace.pl ./$network_name ./$op_directory/$Operations_file_name $word $batch $output_file ./config/multipliers/$multipliers_file_name $baseline_config $comparison_config\n";
		system("./Acc_results_compare_trace.pl ./$network_name ./$op_directory/$Operations_file_name $word $batch $output_file ./config/multipliers/$multipliers_file_name $baseline_config $comparison_config");
	}
}
