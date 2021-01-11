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
use List::Util qw(first);
my $schedule_file=$ARGV[0];
my $opfile=$ARGV[1];
my $net_type=$ARGV[2];
my $array_index=0;
my @layer_names=();
my @num_ops=();
my @Keywords=("time","unit_size","access","to","cost","ti","tb","orders","size","part_lprev","part_lcurr","total_nhops","unit_nhops","part","total","unit");
my $checklayernames=0;
open my $file_pointer, '<', $schedule_file or die $!;
while (my $line= <$file_pointer>){
	if (index($line,"\"ops\"")>=0){
		 my ($num_ops_extracted)= $line=~ /([0-9]+)/;
		 $num_ops[$array_index]=$num_ops_extracted;
		 $array_index++;
	 }
	 elsif (index($line,"\"mappings\"")>=0){
		 $checklayernames=1;
	 }
	 elsif (index($line,"\"")>=0 && $checklayernames==1){
		 my($extracted_name)= $line=~/([a-zA-Z][a-zA-Z0-9_]+)/;
		 if(!first { $_ eq $extracted_name} @Keywords){
			 $layer_names[$array_index]=$extracted_name;
		 }
	 }
}
close ($file_pointer);

if($net_type>=2){
my $checklayernames=0;
open my $file_pointer, '<', $schedule_file."_1" or die $!;
while (my $line= <$file_pointer>){
	if (index($line,"\"ops\"")>=0){
		 my ($num_ops_extracted)= $line=~ /([0-9]+)/;
		 $num_ops[$array_index]=$num_ops_extracted;
		 $array_index++;
	 }
	 elsif (index($line,"\"mappings\"")>=0){
		 $checklayernames=1;
	 }
	 elsif (index($line,"\"")>=0 && $checklayernames==1){
		 my($extracted_name)= $line=~/([a-zA-Z][a-zA-Z0-9_]+)/;
		 if(!first { $_ eq $extracted_name} @Keywords){
			 $layer_names[$array_index]=$extracted_name;
		 }
	 }
}
close ($file_pointer);
}


if($net_type>=3){
my $checklayernames=0;
open my $file_pointer, '<', $schedule_file."_2" or die $!;
while (my $line= <$file_pointer>){
	if (index($line,"\"ops\"")>=0){
		 my ($num_ops_extracted)= $line=~ /([0-9]+)/;
		 $num_ops[$array_index]=$num_ops_extracted;
		 $array_index++;
	 }
	 elsif (index($line,"\"mappings\"")>=0){
		 $checklayernames=1;
	 }
	 elsif (index($line,"\"")>=0 && $checklayernames==1){
		 my($extracted_name)= $line=~/([a-zA-Z][a-zA-Z0-9_]+)/;
		 if(!first { $_ eq $extracted_name} @Keywords){
			 $layer_names[$array_index]=$extracted_name;
		 }
	 }
}
close ($file_pointer);
}


if($net_type>=4){
my $checklayernames=0;
open my $file_pointer, '<', $schedule_file."_3" or die $!;
while (my $line= <$file_pointer>){
	if (index($line,"\"ops\"")>=0){
		 my ($num_ops_extracted)= $line=~ /([0-9]+)/;
		 $num_ops[$array_index]=$num_ops_extracted;
		 $array_index++;
	 }
	 elsif (index($line,"\"mappings\"")>=0){
		 $checklayernames=1;
	 }
	 elsif (index($line,"\"")>=0 && $checklayernames==1){
		 my($extracted_name)= $line=~/([a-zA-Z][a-zA-Z0-9_]+)/;
		 if(!first { $_ eq $extracted_name} @Keywords){
			 $layer_names[$array_index]=$extracted_name;
		 }
	 }
}
close ($file_pointer);
}

if($net_type>=5){
my $checklayernames=0;
open my $file_pointer, '<', $schedule_file."_4" or die $!;
while (my $line= <$file_pointer>){
	if (index($line,"\"ops\"")>=0){
		 my ($num_ops_extracted)= $line=~ /([0-9]+)/;
		 $num_ops[$array_index]=$num_ops_extracted;
		 $array_index++;
	 }
	 elsif (index($line,"\"mappings\"")>=0){
		 $checklayernames=1;
	 }
	 elsif (index($line,"\"")>=0 && $checklayernames==1){
		 my($extracted_name)= $line=~/([a-zA-Z][a-zA-Z0-9_]+)/;
		 if(!first { $_ eq $extracted_name} @Keywords){
			 $layer_names[$array_index]=$extracted_name;
		 }
	 }
}
close ($file_pointer);
}

open $file_pointer, '>', $opfile or die $!;
print $file_pointer "\% NNops =(\n";
for (my $i=0;$i<$array_index;$i++){
	print $file_pointer $layer_names[$i],"=>",$num_ops[$i],",\n";
}
print $file_pointer ");\n1;\n";
close ($file_pointer);
