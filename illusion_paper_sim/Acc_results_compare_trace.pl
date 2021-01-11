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
use List::Util 'sum';
#use IPC::System::Simple qw(system capture);
use File::Basename;
use FileHandle;
use Cwd;
my $top = $ARGV[0];
require $ARGV[1];
our(%NNops, %NNops1, %NNops2, %NNops3, %NNops4);
my $word_size=$ARGV[2];
my $batch_size=$ARGV[3];
require $ARGV[5];
my $baseline_config=$ARGV[6];
my $comparison_config=$ARGV[7];
our(%multipliers);
my $Config_BaseDir="./config/tech"; #TODO: Fill with the correct path
my @arch=($baseline_config, $comparison_config);
my @Freq=(1,1); #TODO: Feed them as inputs later
my $availtypes=2;
my $config_path;
my $nops;my $temp1;my $temp2;
my $output_file= "./results/".$ARGV[4];
my @Summary=([0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]);
open my $file_parser,'>',$output_file or die $!;
# printing variable names
my ($par_dir, $sub_dir);
for (my $systype=0;$systype<$availtypes;$systype++){
	my @Total_output=(0) x 11; #8 entries from the parser
	$config_path=$Config_BaseDir."/Config_".$arch[$systype]."_28nm_".$Freq[$systype]."_".$word_size.".pl";
	print $file_parser $arch[$systype],"\n";;
	print $file_parser "Layer, Active time, Stalled time, Total time, Compute active energy, Register file energy, Idle Energy, Total Cache Energy, Total Mem Energy, Mem Reads, Mem Writes, Total Energy\n";
	my $top1=$top.'/'.$word_size.'/'.$batch_size.'/'.$arch[$systype];
	opendir($par_dir, $top1);
	my @files = sort { $a cmp $b } readdir($par_dir);
        while (my $sub_folders = shift @files) {
		next if ($sub_folders =~ /^..?$/);  # skip . and ..
		my $path = $top1 . '/' . $sub_folders;
		next unless (-d $path);   # skip anything that isn't a directory 
		opendir($sub_dir, $path);
		while (my $file = readdir($sub_dir)) {
			next unless $file =~ /\.out?$/i;
			my $full_path = $path . '/' . $file;
			#print_file_names($full_path);    
			#if($systype==0){
                        #	$nops=$NNops{basename($path)};
			#}elsif($systype==1){
 			#	$nops=$NNops1{basename($path)}; 
                        #}elsif($systype==2){
 			#	$nops=$NNops2{basename($path)}; 
                        #}elsif($systype==3){
 			#	$nops=$NNops3{basename($path)}; 
                        #}elsif($systype==4){
 			#	$nops=$NNops4{basename($path)}; 
                        #}
                        $nops=$NNops{basename($path)};
                        my $basenamep = basename($path);
                        my $part_index = index($basenamep, "part");
                        if ($part_index != -1) {
                            $basenamep = substr($basenamep, 0, $part_index - 1);
                        }
                        #my $multiplier=$multipliers{basename($path)};
                        my $multiplier=$multipliers{$basenamep};
                        #print " ".$nops." is the value of nops \n";
                        #print "calling ARPS\n";
                        #print "ARPS".$full_path." ".$config_path." ".$nops." ".$multiplier."\n";
                        my $parser_output=`./Acc_results_processing_silent.pl $full_path $config_path $nops $multiplier`;
                        print $parser_output;
                        my @_parsed_numbers=split(/\,/,$parser_output);
			$_parsed_numbers[10]=$_parsed_numbers[3]+$_parsed_numbers[4]+$_parsed_numbers[5]+$_parsed_numbers[6]+$_parsed_numbers[7];
			chomp $_parsed_numbers[9];
			print $file_parser basename($path) ,", ", join(",",@_parsed_numbers),"\n";
			@Total_output= map { $Total_output[$_] + $_parsed_numbers[$_]} 0..10;
		}
		closedir($sub_dir);
	}
	@{$Summary[$systype]}=@Total_output;
	print $file_parser "Total,",join(",",@Total_output),"\n";
	closedir($par_dir);
 	if($systype==0){
        	$temp1=$Summary[$systype][2];        
        	$temp2=$Summary[$systype][10];
	}
	#doing the benefits. Coding is super crude here :). Well who cares !
	$Summary[$systype][0]=$Summary[$systype][0]/$temp1*100;
	$Summary[$systype][1]=$Summary[$systype][1]/$temp1*100;
	$Summary[$systype][2]=$Summary[$systype][2]/$temp1*100;
	
	$Summary[$systype][3]=($Summary[$systype][3]+$Summary[$systype][4])/$temp2*100;
	$Summary[$systype][4]=$Summary[$systype][5]/$temp2*100;
	$Summary[$systype][5]=($Summary[$systype][6]+$Summary[$systype][7])/$temp2*100;
	$Summary[$systype][6]=$Summary[$systype][10]/$temp2*100;
	$Summary[$systype][7]=100/$Summary[$systype][2];
	$Summary[$systype][8]=100/$Summary[$systype][6];
	$Summary[$systype][9]=$Summary[$systype][7]*$Summary[$systype][8];

	print $file_parser "Percentage & Benefits \n";
	print $file_parser "System, Active time, Stalled time, Total time, Compute active energy (compute + reg file), Idle energy, Mem energy (Cache + Mem), Total energy, Exec. time benefits, Energy benefits, EDP\n";
	print $file_parser $arch[$systype],", ",,join(",",@{$Summary[$systype]}),"\n";
}
close $file_parser;
sub print_file_names()
{
	my $file = shift;
	my $fh1 = FileHandle->new($file) 
		or die "ERROR: $!"; #ERROR HERE 
	print("$file\n");
}
