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
sub max ($$) {$_[$_[0]<$_[1]] }
require $ARGV[1];
my $nops=$ARGV[2];
#my $ncompute=256; #TODO: replace this with the actual number used from the schedule file
our ($ncompute,$NCORES,$NUM_CACHE_LEVELS,@NCACHE,$NCONTROLLERS,$leakage_power,$Dynamic_E_per_op,$Frequency,$EPI,$Idle_ratio,@CACHE_E,@CACHE_LKG,@MEM_E_per_bit,$MEM_LKG,$LINE_SIZE,$BUF_E,$BUF_LKG,$BUF_LOOKUP,$reg_bit_width,$reg_e_per_bit);
#print $NCORES, "\n";
#Architecture Params
$nops=$nops*$NCORES;
my $reg_energy_access= $reg_bit_width * $reg_e_per_bit;
my$EPI=$Dynamic_E_per_op; #The energy per instruction value in nJ. This value is obtained by dividing the peak power with the max fequency
my$Leakage=$leakage_power/$Frequency;
my $Idle_ratio=($leakage_power/$Frequency)/$EPI; # the percentage of peak energy spent in Idle mode
my $sleep_ratio=($leakage_power/$Frequency)/$EPI; # the percentage of energy spent in contention stall mode
my @Core_cycles= (0) x $NCORES;
my $cycles_cnt=0;
my @Core_ccycles= (0) x $NCORES;
my $ccycles_cnt=0;
my @Core_inst=(0) x $NCORES;
my $inst_cnt=0;
my $TOTAL_cycles=0;
my @fhGETS=();
my @fhGETX=();
my @hGETS=();
my @hGETX=();
my @mGETS=();
my @mGETXIM= ();
my @mGETXSM=();
my @PUTS= ();
my @PUTX=();
my @INV=();
my @INVX= ();
my @MemRD= (0) x $NCONTROLLERS;
my @MemWT= (0) x $NCONTROLLERS;
my @Buffer=(0) x $NCONTROLLERS;
my @MemeACTPRE = (0) x $NCONTROLLERS;
my @MemeRDWR = (0) x $NCONTROLLERS;
my @MemeREF = (0) x $NCONTROLLERS;
my @MemeBKGD = (0) x $NCONTROLLERS;
my $MemEnergy=0;
my $MEM_CNTR=0;
my @Total_mem_power= (0) x $NCONTROLLERS;
my $MEM_PWR_CNTR=0;
my $FILE_NAME=$ARGV[0];
open my $DATA_FILE, '<', $FILE_NAME or die $!;
while (my $line= <$DATA_FILE>){
	if (index($line,"cycles")>=0){
		my ($Num_cycles)= $line=~ /([0-9]+)/;
		$Core_cycles[$cycles_cnt]=$Num_cycles;
		if($TOTAL_cycles<$Num_cycles){
			$TOTAL_cycles=$Num_cycles;
		}
		$cycles_cnt++;
	}
	elsif (index($line,"cCycles")>=0){
		my ($Num_cCycles)= $line=~ /([0-9]+)/;
		$Core_ccycles[$ccycles_cnt]=$Num_cCycles;
		$ccycles_cnt++;
	}
	elsif (index($line,"ops")>=0){
		my ($Num_instrs)= $line=~ /([0-9]+)/;
		$Core_inst[$inst_cnt]=$Num_instrs;
		$inst_cnt++;
	}
	elsif (index($line,"fhGETS")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$fhGETS[++$#fhGETS]=$Num_TMP;
	}
	elsif (index($line,"fhGETX")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$fhGETX[++$#fhGETX]=$Num_TMP;
	}
	elsif (index($line,"hGETS")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$hGETS[++$#hGETS]=$Num_TMP;
	}
	elsif (index($line,"hGETX")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$hGETX[++$#hGETX]=$Num_TMP;
	}
	elsif (index($line,"mGETS")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$mGETS[++$#mGETS]=$Num_TMP;
	}
	elsif (index($line,"mGETXIM")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$mGETXIM[++$#mGETXIM]=$Num_TMP;
	}
	elsif (index($line,"mGETXSM")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$mGETXSM[++$#mGETXSM]=$Num_TMP;
	}
	elsif (index($line,"PUTS")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$PUTS[++$#PUTS]=$Num_TMP;
	}
	elsif (index($line,"PUTX")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$PUTX[++$#PUTX]=$Num_TMP;
	}
	elsif (index($line,"INV")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$INV[++$#INV]=$Num_TMP;
	}
	elsif (index($line,"INVX")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$INVX[++$#INVX]=$Num_TMP;
	}
	elsif (index($line,"rd:")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$MemRD[$MEM_CNTR]=$Num_TMP;
	}
	elsif (index($line,"BufWrites:")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$Buffer[$MEM_CNTR]=$Num_TMP;
	}
	elsif (index($line,"wr:")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$MemWT[$MEM_CNTR]=$Num_TMP;
		$MEM_CNTR++;
	}
        elsif (index($line,"eACTPRE:")>=0){
		$MEM_CNTR--;
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$MemeACTPRE[$MEM_CNTR]=$Num_TMP;
	}
        elsif (index($line,"eRDWR:")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$MemeRDWR[$MEM_CNTR]=$Num_TMP;
	}
        elsif (index($line,"eREF:")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$MemeREF[$MEM_CNTR]=$Num_TMP;
	}
        elsif (index($line,"eBKGD:")>=0){
		my ($Num_TMP)= $line=~ /([0-9]+)/;
		$MemeBKGD[$MEM_CNTR]=$Num_TMP;
		$MEM_CNTR++;
	} 
	elsif (index($line,"Total average power")>=0){
			my ($Num_TMP)= $line=~ /([0-9]+)/;
			$Total_mem_power[$MEM_PWR_CNTR]=$Num_TMP;
			$MEM_PWR_CNTR++;
	}
	elsif (index($line,"eACTPRE:")>=0){
			my ($Num_TMP)= $line=~ /([0-9]+)/;
			$MemEnergy+=$Num_TMP;
	}
        elsif (index($line,"eRDWR:")>=0){
			my ($Num_TMP)= $line=~ /([0-9]+)/;
			$MemEnergy+=$Num_TMP;
	}
        elsif (index($line,"eREF:")>=0){
			my ($Num_TMP)= $line=~ /([0-9]+)/;
			$MemEnergy+=$Num_TMP;
	}
        elsif (index($line,"eBKGD:")>=0){
			my ($Num_TMP)= $line=~ /([0-9]+)/;
			$MemEnergy+=$Num_TMP;
	} 
}

#Total Statitics 
#Timing
my $active_time=$nops/($ncompute*$NCORES)/$Frequency;
my $sum_of_cycles=sum(@Core_cycles);
@Core_cycles=map {$Core_cycles[$_]-$Core_ccycles[$_]}0..$NCORES-1;
#CORE
my @Idle_cycles= map {$Core_cycles[$_]-$Core_inst[$_]} 0 .. $NCORES-1;
my @CPI= map {$Core_cycles[$_]/max($Core_inst[$_],1)} 0 .. $NCORES-1;
my @IPC= sum(@Core_inst)/$TOTAL_cycles;
my @Energy_Cores_Active=map{$Core_inst[$_]*$EPI} 0 .. $NCORES-1;
my @Energy_Cores_IDLE=map{$Core_cycles[$_]*$Leakage} 0 .. $NCORES-1;
my @Energy_Cores_SLEEP=map{$Core_ccycles[$_]*$Idle_ratio*$EPI} 0 .. $NCORES-1;
my $TOTAL_CORES_ACTIVE=sum(@Energy_Cores_Active);
my $TOTAL_CORES_IDLE=sum(@Energy_Cores_IDLE);
my $TOTAL_CORES_SLEEP=sum(@Energy_Cores_SLEEP);
my $AVG_CPI=sum(@CPI)/$NCORES;
my $AVG_IPC=sum(@IPC)/$NCORES;
my $AVG_INSTR=sum(@Core_inst)/$NCORES;
#CACHE
my $total_num_cache=sum(@NCACHE);
my @L1d_Hits=map {$hGETS[$_]+$hGETX[$_]+$fhGETS[$_]+$fhGETX[$_]} 0..$NCORES-1;
my @L1d_Misses=map {$mGETS[$_]+$mGETXSM[$_]+$mGETXIM[$_]} 0..$NCORES-1;
my @L1dunfilterd_Hits=map {$hGETS[$_]+$hGETX[$_]} 0..$NCORES-1;
my @L1I_Hits=map {$hGETS[$_]+$hGETX[$_]+$fhGETS[$_]+$fhGETX[$_]} $NCORES..2*$NCORES-1;
my @L1I_Misses=map {$mGETS[$_]+$mGETXSM[$_]+$mGETXIM[$_]} $NCORES..2*$NCORES-1;
my @Lower_Level_Hits= map {$hGETS[$_]+$hGETX[$_]} 2*$NCORES..$total_num_cache-1;
my @Lower_Level_Misses=map {$mGETS[$_]+$mGETXSM[$_]+$mGETXIM[$_]} 2*$NCORES..$total_num_cache-1;
my @NUM_READS_L1i=map{$L1I_Hits[$_]+$L1I_Misses[$_]}0..$NCORES-1;
my @NUM_WRITES_L1i=map{$L1I_Misses[$_]}0..$NCORES-1+map{$PUTX[$_]+$PUTS[$_]}$NCORES..2*$NCORES-1;;
#my @NUM_WRITES_L1i= @NUM_WRITES_L1i1+map{$PUTX[$_]+$PUTS[$_]}$NCORES..2*$NCORES-1;
my @NUM_READS_L1d=map{$L1d_Hits[$_]+$L1d_Misses[$_]}0..$NCORES-1;
my @NUM_WRITES_L1d=map{$L1d_Misses[$_]}0..$NCORES-1+map{$PUTX[$_]+$PUTS[$_]}0..$NCORES-1;
#my @NUM_WRITES_L1d= @NUM_WRITES_L1d1+map{$PUTX[$_]+$PUTS[$_]}0..$NCORES-1;
my @NUM_READS_LLC=map{$Lower_Level_Hits[$_]+$Lower_Level_Misses[$_]}0..$#Lower_Level_Misses;
my @NUM_WRITES_LLC1=map{$Lower_Level_Misses[$_]}0..$#Lower_Level_Misses;
my @NUM_WRITES_LLC2= map{$PUTX[$_]+$PUTS[$_]}2*$NCORES..$total_num_cache-1;
my @NUM_WRITES_LLC= @NUM_WRITES_LLC1+map{$PUTX[$_]+$PUTS[$_]}2*$NCORES..$total_num_cache-1;
# This a for loop to get the total energy values
my @L1I_ENERGY=map{(($NUM_READS_L1i[$_]+$NUM_WRITES_L1i[$_])*$CACHE_E[0])+$CACHE_LKG[0]*1e-3*($TOTAL_cycles/$Frequency*1e-9)}0..$NCORES-1;
my @L1d_ENERGY=map{(($NUM_READS_L1d[$_]+$NUM_WRITES_L1d[$_])*$CACHE_E[1])+$CACHE_LKG[1]*1e-3*($TOTAL_cycles/$Frequency*1e-9)}0..$NCORES-1;
my @LLC_ENERGY=();
for (my $i=1;$i<=$#NCACHE;$i++){
        for(my $j=0;$j<$NCACHE[$i];$j++){
                my $index=($i-1)*$NCACHE[$i]+$j;
                $LLC_ENERGY[$index]=($NUM_READS_LLC[$index]+$NUM_WRITES_LLC[$index])*$CACHE_E[$i+1]+$CACHE_LKG[$i+1]*1e-3*($TOTAL_cycles/$Frequency*1e-9)*$NCACHE[$i];
        }
}
my @L1I_HitRate=map{$L1I_Hits[$_]/max(1,($L1I_Hits[$_]+$L1I_Misses[$_]))}0..$#L1I_Hits;

my @L1d_HitRate=map{$L1d_Hits[$_]/max(1,($L1d_Hits[$_]+$L1d_Misses[$_]))}0..$#L1d_Hits;
my @LLC_HitRate=map{$Lower_Level_Hits[$_]/max(1,($Lower_Level_Hits[$_]+$Lower_Level_Misses[$_]))}0..$#Lower_Level_Hits;
my $TOTAL_L1i_ENERGY=sum(@L1I_ENERGY);
my $TOTAL_L1d_ENERGY=sum(@L1d_ENERGY);
my $TOTAL_LLC_ENERGY=sum(@LLC_ENERGY);
my $L1I_AVG_HIT_RATE=sum(@L1I_HitRate)/scalar(@L1I_HitRate);
my $L1d_AVG_HIT_RATE=sum(@L1d_HitRate)/scalar(@L1d_HitRate);
my @L1d_MPKI=map{$L1d_Misses[$_]/$Core_inst[$_]*1e3}0 ..$NCORES-1;
my @L1I_MPKI=map{$L1I_Misses[$_]/$Core_inst[$_]*1e3}0 ..$NCORES-1;
my $L2_MPKI=0;
for(my $i=$NCACHE[1];$i<$NCACHE[1]+$NCACHE[2];$i++){
	$L2_MPKI += @Lower_Level_Misses[$i];
}
$L2_MPKI= $L2_MPKI*1e3/sum(@Core_inst);
my $LLC_MPKI=sum(@Lower_Level_Misses)*1e3/sum(@Core_inst);
my $L1I_AVG_MPKI=sum(@L1I_MPKI)/scalar(@L1I_MPKI);
my $L1d_AVG_MPKI=sum(@L1d_MPKI)/scalar(@L1d_MPKI);
my $L1d_unfiltered=sum(@L1dunfilterd_Hits)/scalar(@L1dunfilterd_Hits);



#Memory
my $MEM_RD=sum(@MemRD);
my $MEM_WT=sum(@MemWT);
my @MEM_RD_ENERGY=map{$MemRD[$_]*$MEM_E_per_bit[0]*$LINE_SIZE}0..$NCONTROLLERS-1;
my @MEM_WT_ENERGY=map{$MemWT[$_]*$MEM_E_per_bit[1]*$LINE_SIZE}0..$NCONTROLLERS-1;
my @MEM_ACTPRE_ENERGY=map{$MemeACTPRE[$_]}0..$NCONTROLLERS-1;
my @MEM_RDWR_ENERGY=map{$MemeRDWR[$_]}0..$NCONTROLLERS-1;
my @MEM_REF_ENERGY=map{$MemeREF[$_]}0..$NCONTROLLERS-1;
my @MEM_BKGD_ENERGY=map{$MemeBKGD[$_]}0..$NCONTROLLERS-1;
my $TOTAL_RD_ENERGY=sum(@MEM_RD_ENERGY);
my $TOTAL_WT_ENERGY=sum(@MEM_WT_ENERGY);
my $TOTAL_ACTPRE_ENERGY=sum(@MEM_ACTPRE_ENERGY);
my $TOTAL_RDWR_ENERGY=sum(@MEM_RDWR_ENERGY);
my $TOTAL_REF_ENERGY=sum(@MEM_REF_ENERGY);
my $TOTAL_BKGD_ENERGY=sum(@MEM_BKGD_ENERGY);
my $TOTAL_MEM_ENERGY = $TOTAL_ACTPRE_ENERGY+$TOTAL_RDWR_ENERGY+$TOTAL_REF_ENERGY+$TOTAL_BKGD_ENERGY;
if($TOTAL_MEM_ENERGY==0){
   $TOTAL_MEM_ENERGY = $TOTAL_RD_ENERGY+$TOTAL_WT_ENERGY+$MEM_LKG*1e-3*($TOTAL_cycles/$Frequency*1e-9);
}

#Statistics printin
#print "Total cycles", $sum_of_cycles,"\n";
#print "Leakage", $EPI*$Idle_ratio,"\n";
#print "Total Leakage", $sum_of_cycles*$EPI*$Idle_ratio,"\n";
#print "Total dynamic", sum(@Core_inst)*$EPI*(1-$Idle_ratio),"\n";
#print "MAximum Cycles", $TOTAL_cycles,"\n";
#print "Total # instructions", sum(@Core_inst),"\n";
#print "Total Instruction Cache accesses", sum(@NUM_READS_L1i)+sum(@NUM_WRITES_L1i), "\n";
#print "Total Data Cache accesses", sum(@NUM_READS_L1d)+sum(@NUM_WRITES_L1d),"\n";
#print "Total L2 Cache reads", sum(@NUM_READS_LLC),"\n";
#print "Total L2 Cache writes", sum(@NUM_WRITES_LLC2),"\n";
#print "Total Memory accesses", sum(@MemRD)+sum(@MemWT),"\n";
#print "Execution time:",$TOTAL_cycles/$Frequency*1e-9," (s)\n";
#print "--Precentage of time in Execution:",sum(@Core_inst)/$sum_of_cycles*100,"%\n";
#print "--Percentage of time Waiting:",sum(@Idle_cycles)/$sum_of_cycles*100,"%\n";
#print "--Percentage of time stalled:",sum(@Core_ccycles)/$sum_of_cycles*100,"%\n";
#print "The Average CPI:",$AVG_CPI,"\n";
#print "The Average IPC:",$AVG_IPC,"\n";
#print "CORE Energy:",($TOTAL_CORES_ACTIVE+$TOTAL_CORES_IDLE+$TOTAL_CORES_SLEEP)*1e-9, "(J)\n";
#print "--Active Energy                        :",$TOTAL_CORES_ACTIVE*1e-9,"(J)\n";
#print "--IDLE Energy                          :",$TOTAL_CORES_IDLE*1e-9,"(J)\n";
#print "--Sleep Energy due to contention stalls:",$TOTAL_CORES_SLEEP*1e-9,"(J)\n";
#print "CACHE Energy:",($TOTAL_L1i_ENERGY+$TOTAL_L1d_ENERGY+$TOTAL_LLC_ENERGY)*1e-9,"(J)\n";
#print "--L1 Instruction Energy:",$TOTAL_L1i_ENERGY*1e-9,"(J)\n";
#print "--L1 Data Energy       :",$TOTAL_L1d_ENERGY*1e-9,"(J)\n";
#print "--From L2 to LLC Energy:",$TOTAL_LLC_ENERGY*1e-9,"(J)\n";
#print "HitRates: \n";
#print "--L1I:",$L1I_AVG_HIT_RATE*100,"% \n";
#print "----unfiltered:",$L1d_unfiltered,"\n";
#print "--L1d:",$L1d_AVG_HIT_RATE*100,"% \n";
#print "Lower Levels:",$#LLC_HitRate,"\n";
#print "$_\n" for @LLC_HitRate;
#print "MPKI :\n";
#print "--l1I",$L1I_AVG_MPKI,"\n";
#print "--L1d",$L1d_AVG_MPKI,"\n";
#print "Number of lower levels", $#Lower_Level_Misses,"\n";
#print "Lower Levels:",$LLC_MPKI,"\n";
#print "--L2:",$L2_MPKI,"\n";


#print "Memory Reads:", sum(@MemRD),"\n";
#print "Memory Writes:", sum(@MemWT),"\n";

#if($Total_mem_power[0]==0){
#	print "Memory Energy:",($TOTAL_RD_ENERGY+$TOTAL_WT_ENERGY)*1e-12+$MEM_LKG*1e-3*$TOTAL_cycles/$Frequency*1e-9,"(J)\n";
#	print "-- Read Energy:",$TOTAL_RD_ENERGY*1e-12,"(J)\n";
#	print "--Write Energy:",$TOTAL_WT_ENERGY*1e-12,"(J)\n";
#}else{
#	print "Memory Energy:",(sum(@Total_mem_power)*$TOTAL_cycles/$Frequency*1e-9)*1e-3,"(J)\n";
#}
#print "Buffer Writes:", sum(@Buffer),"\n";
#print "Buffer Energy:", ((sum(@Buffer)*$BUF_E)*1e-9+((sum(@MemRD)+sum(@MemWT))*$BUF_LOOKUP)*1e-9+$BUF_LKG*$NCONTROLLERS*1e-3*$TOTAL_cycles/$Frequency*1e-9),"(J)\n";

my $Exec_TIme=$TOTAL_cycles/$Frequency*1e-9;
print $active_time*1e-9*2,",",$Exec_TIme-$active_time*1e-9,",",$Exec_TIme+$active_time*1e-9,",",$nops*$EPI*1e-9,",",$nops*$reg_energy_access*2e-12,",",($Exec_TIme+$active_time*1e-9)*$leakage_power*$NCORES,",",($TOTAL_L1i_ENERGY+$TOTAL_L1d_ENERGY+$TOTAL_LLC_ENERGY)*1e-9,",",($TOTAL_MEM_ENERGY)*1e-12,",",sum(@MemRD),",",sum(@MemWT),"\n";
