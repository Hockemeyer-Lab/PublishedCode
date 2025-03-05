import os
import os.path
import pyarrow as pa
import pyarrow.csv as csv
import pandas as pd
import bisect
from zipfile import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib_venn import venn3
import numpy as np
import re
import seaborn as sns
from time import process_time as pt
from Bio.Seq import Seq
from Bio import pairwise2
from matplotlib.colors import LogNorm, Normalize
from matplotlib.colors import LinearSegmentedColormap
import sys
import statistics
from tqdm import tqdm 
import swifter
import glob
from itertools import zip_longest
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cairosvg
import math
from matplotlib import cm
import colormaps as cmaps

def rev_comp(s): #find the reverse complement of a DNA sequence
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    reverse_complement = "".join(complement.get(base, base) for base in reversed(s))
    return reverse_complement

def unique(list1): #return only the unique elements of a list (no duplicates)
    x = np.array(list1)
    unique_list = np.unique(x).tolist()
    return unique_list

def sgFinder(seq,wt): # find the sgRNA cutsite in an amplicon
    if seq in wt:
        return [wt.find(seq) +16, wt.find(seq) +17]
    else:
        seq = rev_comp(seq)
        return [wt.find(seq) +2, wt.find(seq) +3]

def splicing_intact(row): #convert splicing info to bool
    start = row['intact_splice_start']
    end = row['intact_splice_end']
    
    if start < 0 or end < 0:
        return False
    else:
        return True

def translate(sequence):
    DNA_Codons = {
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TGT": "C", "TGC": "C",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TTT": "F", "TTC": "F",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    "CAT": "H", "CAC": "H",
    "ATA": "I", "ATT": "I", "ATC": "I",
    "AAA": "K", "AAG": "K",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATG": "M",
    "AAT": "N", "AAC": "N",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TGG": "W",
    "TAT": "Y", "TAC": "Y",
    "TAA": "_", "TAG": "_", "TGA": "_"
    }
    parsed = [sequence[i:i+3] for i in range(0,len(sequence),3)]
    translated = [DNA_Codons[x.upper()] for x in parsed]
    return ''.join(translated)

def interpret_guides(gene_seq,exon_seqs_df,sgRNA_list): #if there is no HDR file provided, define regions of interest to interpret around sgRNA cutsites
    sgRNA_list = sgRNA_list.rename(columns={"sgRNA#": "sgRNA # oPool Name", 'sgRNA sequence' :'sgRNA seq'})
    orientations = []
    regions = []
    exons = []
    for index, row in sgRNA_list.iterrows():
        row['sgRNA seq'] = row['sgRNA seq'].upper()
        if row['sgRNA seq'] in gene_seq:
            cutsite = gene_seq.find(row['sgRNA seq']) + 17
            orientations.append('F')
        elif rev_comp(row['sgRNA seq']) in gene_seq:
            cutsite = gene_seq.find(rev_comp(row['sgRNA seq'])) + 3
            orientations.append('R')
        else:
            print('sgRNA #%i not found in gene sequence. Please double-check inputs and retry' % row['sgRNA # oPool Name'])
            sys.exit()
        amplicon_info = exon_seqs_df[exon_seqs_df["Amplicon #"] == row['Amplicon']]
        exons.append(int(amplicon_info['Exon # ']))
        exon_start = gene_seq.find(amplicon_info['Exon Sequence'].item())
        exon_end = exon_start + len(amplicon_info['Exon Sequence'].item())
        rel_start = cutsite - 20
        rel_end = cutsite + 20
        if rel_start < exon_start:
            rel_start = exon_start
        if rel_end > exon_end:
            rel_end = exon_end
        regions.append(gene_seq[rel_start:rel_end])
    sgRNA_list['sgRNA orientation? F or R'] = orientations
    sgRNA_list['Exon #'] = exons
    sgRNA_list['region around sgRNA that is exon'] = regions
    return sgRNA_list

def get_partials(wt_HDR_template,tagged_HDR_template,wt_seq,mini_seq,amp_exon_seq,sg_seq,protdnaseq,amp_seq): #make a list of partial HDR templates which may result from incomplete HDR (only one side of cutsite)
    #account for any mini-seqs that we have made longer to allow for HDR identification
    pHDR_L = ''
    pHDR_R = ''

    if len(wt_seq) != len(mini_seq):
        wt_seq = wt_HDR_template[tagged_HDR_template.find(mini_seq):tagged_HDR_template.find(mini_seq)+len(mini_seq)]
    
    #First determine where the cutsite is relative to the HDR template and the region of interest
    if sg_seq in wt_HDR_template:
        cutsite = wt_HDR_template.find(sg_seq) + 17 - tagged_HDR_template.find(mini_seq)
    elif rev_comp(sg_seq) in wt_HDR_template:
        cutsite = wt_HDR_template.find(rev_comp(sg_seq)) + 3 - tagged_HDR_template.find(mini_seq)
    else:
        print('Warning: sgRNA %i sequence not found in HDR template region, partial HDR not included' % sg_seq)
        return(pHDR_L,pHDR_R)

    #If the cutsite is outside the region of interest, no partial HDR is possible
    if cutsite <0 or cutsite>len(wt_seq):
        return(pHDR_L,pHDR_R)
    
    #find the translation frame of the region of interest to determine which codons may be split by the cutsite
    frame_cod = (protdnaseq.find(amp_exon_seq) + amp_exon_seq.find(wt_seq))%3
    if frame_cod >0:
        frame_cod = 3-frame_cod
    parsed_wt = [wt_seq[i:i+3] for i in range(frame_cod,len(wt_seq),3)]
    position_nums =  [[i,i+1,i+2] for i in range(frame_cod,len(wt_seq),3)]
    parsed_mut = [mini_seq[i:i+3] for i in range(frame_cod,len(wt_seq),3)]
    changed_codons = [position_nums[i] for i  in [i for i, val in enumerate(parsed_wt) if parsed_wt[i] != parsed_mut[i]]]
    
    #split codons by cutsite and see if there are changes on both sides of the cutsite
    #first check to see if the cutsite is in a codon that is mutated
    leftP = ''
    rightP = ''
    wt_L = ''
    wt_R = ''
    
    centers = any(num < cutsite for codon in changed_codons for num in codon) and any(num >= cutsite for codon in changed_codons for num in codon)
    if centers: #if so, make sure that partial HDR events preserve the entire cut codon
        leftP = mini_seq[:cutsite+frame_cod]
        rightP = mini_seq[cutsite-(3-frame_cod):]
        wt_L = wt_seq[:cutsite+frame_cod]
        wt_R = wt_seq[cutsite-(3-frame_cod):]
    else:
        leftP = mini_seq[:cutsite]
        rightP = mini_seq[cutsite:]
        wt_L = wt_seq[:cutsite]
        wt_R = wt_seq[cutsite:]
    
    if leftP!='' and rightP!='' and wt_L != '' and wt_R != '':
        #check to see if there are at least two mutations on both left and right and that they are not in the WT sequence
        if len([i for i in range(len(leftP)) if leftP[i] != wt_L[i]]) >= 2:
            leftP = leftP.replace('-','')
            if leftP not in amp_seq:
                pHDR_L = leftP
        if len([i for i in range(len(rightP)) if rightP[i] != wt_R[i]]) >= 2 and rightP not in amp_seq:
            rightP = rightP.replace('-','')
            if rightP not in amp_seq:
                pHDR_R = rightP
        
    return(pHDR_L,pHDR_R)

def identify_relevant_HDR(HDR_lib_design,sgRNA_df,amplicon_seq,protdnaseq): #identify which HDR templates are relevant for a specific amplicon and call partials
    rel_HDR = HDR_lib_design.copy()
    rel_HDR = rel_HDR[rel_HDR["sgRNA # oPool Name"].isin(sgRNA_df["sgRNA # oPool Name"].values.tolist())].reset_index() 
    collapsed_mini_seqs = rel_HDR['mini_seqs'].apply(lambda x: x.replace('---',''))
    #confirm that the HDR sequences are not in the WT allele
    collapsed_mini_seqs = [i  if i not in amplicon_seq else '' for i in collapsed_mini_seqs]
    rel_HDR['collapsed_mini_seqs'] = collapsed_mini_seqs
    #if the HDR sequence is in the WT allele (eg. if a deletion creates a region of interest that is too short to be unique), try expanding the mini-seq window by 10nts
    if '' in collapsed_mini_seqs:
        lost_HDR_ind = [i for i in range(len(collapsed_mini_seqs)) if collapsed_mini_seqs[i] =='']
        for ind in lost_HDR_ind:
            mini = rel_HDR.loc[ind,'mini_seqs']
            mini_ind =rel_HDR.loc[ind,'with_tagging_mutations_for_single_muts'].find(mini)
            new_mini = rel_HDR.loc[ind,'with_tagging_mutations_for_single_muts'][mini_ind-5:mini_ind+len(mini)+5]
            if new_mini.replace('---','') not in amplicon_seq:
                rel_HDR.loc[ind,'mini_seqs'] = new_mini
                rel_HDR.loc[ind,'collapsed_mini_seqs'] = new_mini.replace('---','')
    if '' in rel_HDR['collapsed_mini_seqs'].tolist():
        print('%i HDR events are unidentifiable in the amplicon. Check for repetitive sequences' % sum([1 for x in rel_HDR['collapsed_mini_seqs'].tolist() if x =='']))
    
    #get relevant partial HDR templates which have at least two identifiable tagging mutations; framing of codon: may get squirrely around intronic silent mutations
    left_HDRp = []
    right_HDRp = []
    all_partials = rel_HDR.apply(lambda x: get_partials(x['HDR window'],x['with_tagging_mutations_for_single_muts'],x['region of interest prechange'],x['mini_seqs'],x['which part of HDR window is exon'],x['sgRNA seq'],protdnaseq,amplicon_seq),axis=1)
    left_HDRp = [x[0] for x in all_partials]
    right_HDRp = [x[1] for x in all_partials]
    rel_HDR['Left_Partial_HDR'] = left_HDRp
    rel_HDR['Right_Partial_HDR'] = right_HDRp
    return rel_HDR

def validate_subs(aligned,ref,rel_HDR,cutsite): ################################################# Does Not Handle Insertions well! Need to fix that if we want to look at insertions ########################################################
    mismatch = [i for i, val in enumerate(aligned) if val != ref[i]]
    mismatch_name = "_".join(str(m) for m in mismatch) #this creates a string for identifying the mutations

    if mismatch_name == "": #this is the WT sequence
        return 'WT',aligned,"","","", False
    ################################################# Does Not Handle Insertions well! Need to fix that if we want to look at insertions ########################################################
    else:  
        #First check to see if there is full HDR
        collapsed_align = aligned.replace("-","")
        lst_full_HDR = [mini_seq for mini_seq in unique(rel_HDR['mini_seqs']) if aligned.find(mini_seq) != -1]
        if len(aligned)==len(collapsed_align) and len(lst_full_HDR) >0:
            possible = []
            for mini_seq_real in lst_full_HDR:
                if aligned.find(mini_seq_real) >0:
                    edited = ref[: aligned.find(mini_seq_real)]+ (mini_seq_real) + ref[aligned.find(mini_seq_real)+len(mini_seq_real):]
                    possible.append(edited)
            if len(unique(possible)) == 1 and "-" not in ref: #if there is only one possible associated HDR event
                return "HDR",possible[0], "HDR","HDR","HDR", True
            elif len(unique([i.replace("-","") for i in possible])) ==1 and "-" not in ref: #if there are multiple possible associated HDR events which all collapse on the same value
                return "HDR",possible[0], "HDR","HDR","HDR", True
            else:
                return "HDR", "chimera", "","","", False #more than one HDR events attempted or amplification template switching
        #then check to see if there are HDR-mediated deletions
        lst_possible_HDR = [mini_seq for mini_seq in unique(rel_HDR['collapsed_mini_seqs']) if collapsed_align.find(mini_seq) != -1]
        if (len(aligned)-len(collapsed_align) ==3) and (len(lst_possible_HDR) >0):
            possible = []
            for mini_seq_real in lst_possible_HDR:
                #sometimes deletions don't align as we would expect or may be generated from multiple cutting events, so use some flexibility in determining deletion events
                if collapsed_align.find(mini_seq_real) >0:
                    temp_HDR_DF = rel_HDR.copy()
                    possible_templates = temp_HDR_DF[temp_HDR_DF['collapsed_mini_seqs']==mini_seq_real]['mini_seqs'].tolist()
                    for template in possible_templates:
                        if '---' in template:    
                            edited = ref[:collapsed_align.find(mini_seq_real)] + template + ref[collapsed_align.find(mini_seq_real)+3+len(mini_seq_real):]
                            if len(ref) == len(edited):
                                possible.append(edited)
                            else: #if a deletion event appears to be caused by HDR but there are apparently more than 3nt deleted
                                print(aligned)
                                print(mini_seq_real) 
            if len(unique(possible)) == 1 and "-" not in ref: #if there is only one possible associated HDR event
                return "HDR",possible[0], "HDR","HDR","HDR", True
            elif len(unique([i.replace("-","") for i in possible])) ==1 and "-" not in ref: #if there are multiple possible associated HDR events which all collapse on the same value
                return "HDR",possible[0], "HDR","HDR","HDR", True
            else:
                return "HDR", "chimera", "","","", False #more than one HDR events attempted or amplification template switching or some sort of insertion detected
            
        #now check to see if there are instead any partial HDR events
        collapsed_align = aligned.replace("-","")
        partial_HDR_list = unique([x for x in rel_HDR['Left_Partial_HDR'].tolist() if x != '']) + unique([x for x in rel_HDR['Right_Partial_HDR'].tolist() if x != ''])
        if len([partial for partial in partial_HDR_list if collapsed_align.find(partial) != -1]) >=1:
            
            lst_partials = [partial for partial in partial_HDR_list if collapsed_align.find(partial) != -1]
            possible = []
            if (len(lst_partials) >= 1):
                for mini_seq_real in lst_partials:
                    if aligned.find(mini_seq_real) >0:
                        edited = ref[: aligned.find(mini_seq_real)]+ (mini_seq_real) + ref[aligned.find(mini_seq_real)+len(mini_seq_real):]
                        if len([i for i,val in enumerate(edited) if val != ref[i]]) >=2: #make sure there are at least two mutations
                            possible.append(edited)
                if len(unique([i.replace("-","") for i in possible])) ==1 and "-" not in ref:
                    return "HDRp", possible[0],"HDRp","HDRp","HDRp", True  
            
            if (len(aligned)-len(collapsed_align)) == 3 and aligned.count("---") ==1:
                if (len(lst_partials) >= 1):
                    possible = []
                    for mini_seq_real in lst_partials:         
                        if collapsed_align.find(mini_seq_real) >0:
                            # again use some flexibility in determining deletion events
                            if mini_seq_real in rel_HDR['Left_Partial_HDR'].tolist():
                                temp_HDR_DF = rel_HDR.copy()
                                possible_templates = temp_HDR_DF[temp_HDR_DF['Left_Partial_HDR']==mini_seq_real]['mini_seqs'].tolist()
                                for template in possible_templates:
                                    if '---' in template:
                                        edited = ref[:collapsed_align.find(mini_seq_real)] + mini_seq_real[:template.find('---')]+'---'+mini_seq_real[template.find('---'):] + ref[collapsed_align.find(mini_seq_real)+3+len(mini_seq_real):]
                                        if len(ref) == len(edited) and edited.count("---") == 1:
                                            possible.append(edited)
                                        else: #if a deletion event appears to be caused by HDR but there are apparently more than 3nt deleted
                                            print(aligned)
                                            print(mini_seq_real) 
                    
                            if mini_seq_real in rel_HDR['Right_Partial_HDR'].tolist():
                                temp_HDR_DF = rel_HDR.copy()
                                possible_templates = temp_HDR_DF[temp_HDR_DF['Right_Partial_HDR']==mini_seq_real]['mini_seqs'].tolist()
                                for template in possible_templates:
                                    if '---' in template:
                                        delsite_from_end = len(template)-template.find('---')-3
                                        edited = ref[:collapsed_align.find(mini_seq_real)] + mini_seq_real[:-delsite_from_end]+'---'+mini_seq_real[-delsite_from_end:] + ref[collapsed_align.find(mini_seq_real)+3+len(mini_seq_real):]
                                        if len(ref) == len(edited) and edited.count("---") == 1:
                                            possible.append(edited)
                                        else: #if a deletion event appears to be caused by HDR but there are apparently more than 3nt deleted
                                            print(aligned)
                                            print(mini_seq_real) 
                    if len(unique(possible)) == 1 and "-" not in ref: #if there is only one possible associated HDR event
                        return "HDRp",possible[0], "HDRp","HDRp","HDRp", True
                    elif len(unique([i.replace("-","") for i in possible])) ==1 and "-" not in ref: #if there are multiple possible associated HDR events which all collapse on the same value
                        return "HDRp",possible[0], "HDRp","HDRp","HDRp", True        
        
        #If not HDR or partial HDR
        #NHEJ               
        if "_" in mismatch_name:
            nhej = [int(x) for x in mismatch_name.split('_')]
        else:
            nhej = [int(mismatch_name)]
        
        ###########################################################################################################################################################################
        ###################################### Need to fix this if there's going to be Insertion analysis #########################################################################
        # #adjust cutsite values if there's an insertion which shifts things over
        # cutsite = potential_cutsite_idx.copy()
        # if "-" in ref:
        #     ins_idx = [i for i, c in enumerate(ref) if c == "-"]
        #     cutsite = [cut + 1 if cut > idx else cut for idx in ins_idx for cut in cutsite]
        
        # group into consecutiveish (within 2bp) stretches to consider as one gene editing event
        i1 = 0
        rels = []
        while i1 < len(nhej):
            i2 = i1
            while i2 < len(nhej)-1 and ( (nhej[i2+1] == (nhej[i2]+1)) or (nhej[i2+1] == (nhej[i2]+2))):
                i2 = 1 + i2
            rels.append(list(range(nhej[i1],nhej[i2]+1)))
            i1 = i2+1

        # check if each stretch is within a cutsite window +/-2 of cutsite
        real = [r for r in rels if any(i in cutsite or i-1 in cutsite or i+1 in cutsite for i in r)]

        # Check for alignment issues with longer deletions
        if len(real) == 1:
            if "-" in aligned:
                dtem = [aligned[i] for i in real[0]]
                if "-" in dtem:
                    PCR_error =  sorted(list(set(nhej).symmetric_difference(set(real[0]))))
                    if len(PCR_error) >0:
                        for i in PCR_error:
                            if aligned[i] == "-":
                                if i in list(range(real[0][0]-10, real[0][0])) or i in list(range(real[0][-1], real[0][-1]+10)):
                                    return "NHEJ", "Alignment_Issue", "","","", False
                                i1 = 0
                                temp = []
                                while i1 < len(PCR_error):
                                    i2 = i1
                                    while i2 < len(PCR_error)-1 and ( (PCR_error[i2+1] == (PCR_error[i2]+1)) or (PCR_error[i2+1] == (PCR_error[i2]+2))):
                                        i2 = 1 + i2
                                    temp.append(list(range(PCR_error[i1],PCR_error[i2]+1)))
                                    i1 = i2+1
                                for g in temp:
                                    if len(g) >1:
                                        gtem = [aligned[k] for k in g]
                                        if "-" in gtem:
                                            return "NHEJ", "Alignment_Issue", "","","", False
        # tag each mutation that is within a cutsite window with what it is
        subs = []
        ins =[]
        dels = []
        # check number of potentially real subs
        if len(real) == 0:
            return mismatch_name,ref,"","","",False
        elif len(real) > 1:
            return mismatch_name,"More than one unique sgRNA","","","", False
        elif len(real[0]) == 1: # comment this to turn off "all NHEJ indels/subs of 1 nt to wt seq instead "
            return mismatch_name, ref, "","","", False
        else:
            edited = aligned
            PCR_error =  sorted(list(set(nhej).symmetric_difference(set(real[0]))))
            if len(PCR_error) > 0:
                for i in PCR_error:
                    edited = edited[:i] + ref[i] + edited[i+1:]   
            ins = [i for i in real[0] if edited[i] != "-" and ref[i] == "-" and ref[i] != edited[i]]
            subs = [i for i in real[0] if edited[i] != "-" and ref[i] != "-" and ref[i] != edited[i]]
            dels = [i for i in real[0] if edited[i] == "-" and ref[i] != "-" and ref[i] != edited[i]]
            return mismatch_name, edited,"_".join(map(str,subs)), "_".join(map(str,ins)),"_".join(map(str,dels)),False

def find_frame(row, amp_exon_seq, protdnaseq): # find the nt position in the codon where the first nt change occurred
    exon_frame =  protdnaseq.find(amp_exon_seq)%3
    al = row.Aligned_Sequence_PCR_Error_fixed[row.intact_splice_start+3:row.intact_splice_end+3]
    ref = row.Reference_Sequence[row.intact_splice_start+3:row.intact_splice_end+3]
    first_mismatch = [x for x in range(len(al)) if al[x] != ref[x]][0]
    frame = (first_mismatch%3 + exon_frame)%3
    if frame ==3:
        frame = 0
    return str(frame)

def attempt_name(wt_aa, protein_seq, row, amp_exon_seq, protdnaseq, amp):
    allele_aa = row['translated_where_possible']
    mut_type = ""
    name = ""

    if allele_aa != False and 'False' not in allele_aa:
        num_adjust = protein_seq.find(wt_aa) + 1
        if row['HDR'] == False:
            if row['mismatch_idx'] == 'WT':
                    name = 'WT'
                    mut_type = 'None'
            elif allele_aa != wt_aa:
                shorter = min([len(allele_aa),(len(wt_aa))])
                if '_' in allele_aa and '_' not in wt_aa:
                    mut_pos = [i for i in range(shorter) if wt_aa[i] != allele_aa[i]]
                    mut_type = 'NHEJ Truncation'
                    pos_truncation = allele_aa.find('_') + num_adjust
                    name = 'NHEJ_'+ wt_aa[mut_pos[0]]+str(mut_pos[0]+num_adjust)+allele_aa[mut_pos[0]]+'_trunc'+str(pos_truncation)+'*'
                elif len(wt_aa) - len(allele_aa) == 1:
                    mut_pos = [i for i in range(len(allele_aa)) if wt_aa[i] != allele_aa[i]]
                    if len(mut_pos) > 0:
                        first_mut = mut_pos[0]
                    else:
                        first_mut = -1
                    allele_aa = allele_aa[:first_mut] + '-' + allele_aa[first_mut:]
                    mut_pos = [i for i in range(len(allele_aa)) if wt_aa[i] != allele_aa[i]]
                    if len(mut_pos) == 1:
                        mut_pos = mut_pos[0]
                        mut_type = 'Single Deletion'
                        name = wt_aa[mut_pos] + str(mut_pos+num_adjust) + 'del'
                    else:
                        ins_idx = row['ins_idx']
                        if len(ins_idx) >0:
                            ins_idx = ins_idx.split('_')
                        del_idx = row['del_idx']
                        if len(del_idx) >0:
                            del_idx = del_idx.split('_')
                        d3 = len(ins_idx)-len(del_idx)
                        if d3<0:
                            operator = ''
                        else:
                            operator = '+'
                        frame = find_frame(row,amp_exon_seq,protdnaseq)
                        if d3 % 3 == 0:
                            mut_type = 'In-frame NHEJ'
                            name = 'NHEJ_'+ wt_aa[mut_pos[0]]+str(mut_pos[0]+num_adjust)+ '_' + frame + operator + str(d3) + 'nt'
                        else:
                            mut_type = 'Frame Shift NHEJ'
                            name = 'NHEJ_'+ wt_aa[mut_pos[0]]+str(mut_pos[0]+num_adjust)+ '_' + frame +  operator + str(d3) + 'nt'
                else:
                    mut_pos = [i for i in range(shorter) if wt_aa[i] != allele_aa[i]]
                    ins_idx = row['ins_idx']
                    if len(ins_idx) >0:
                        ins_idx = ins_idx.split('_')
                    del_idx = row['del_idx']
                    if len(del_idx) >0:
                        del_idx = del_idx.split('_')
                    d3 = len(ins_idx)-len(del_idx)  
                    if d3<0:
                        operator = ''
                    else:
                        operator = '+'
                    frame = find_frame(row,amp_exon_seq,protdnaseq)
                    if d3 % 3 == 0:
                        mut_type = 'In-frame NHEJ'
                        name = 'NHEJ_'+ wt_aa[mut_pos[0]]+str(mut_pos[0]+num_adjust)+ '_' + frame +  operator + str(d3) + 'nt'
                    else:
                        mut_type = 'Frame Shift NHEJ'
                        name = 'NHEJ_'+ wt_aa[mut_pos[0]]+str(mut_pos[0]+num_adjust)+ '_' + frame +  operator + str(d3) + 'nt'
        elif len(wt_aa) == len(allele_aa):
            mut_pos = [i for i in range(len(wt_aa)) if wt_aa[i] != allele_aa[i]]
            if len(mut_pos) == 1:
                mut_pos = mut_pos[0]
                if allele_aa[mut_pos] == 'A':
                    mut_type = 'Alanine'
                    name = wt_aa[mut_pos] + str(mut_pos+num_adjust) + allele_aa[mut_pos]
                else:
                    mut_type = 'Substitution'
                    name = wt_aa[mut_pos] + str(mut_pos+num_adjust) + allele_aa[mut_pos]
            elif len(mut_pos) == 0:
                try:
                    transl_start = row['Reference_Sequence'].find(amp_exon_seq)
                    wt = row['Reference_Sequence'][transl_start:transl_start+len(amp_exon_seq)]
                    mut = row['Aligned_Sequence_PCR_Error_fixed'][transl_start:transl_start+len(amp_exon_seq)]
                    frame = protdnaseq.find(wt)%3
                    if frame >0:
                        frame = 3-frame
                        wt = [wt[:frame]]+[wt[x:x+3] for x in range(frame,len(wt),3)]
                        mut = [mut[:frame]]+[mut[x:x+3] for x in range(frame,len(mut),3)]
                    else:
                        wt = [wt[x:x+3] for x in range(frame,len(wt),3)]
                        mut = [mut[x:x+3] for x in range(frame,len(mut),3)]
                    inds = [i for i in range(len(wt)) if wt[i]!=mut[i]]
                    if len(inds) == 1:
                        nt_muts = [i for i in range(0,3) if wt[inds[0]][i]!=mut[inds[0]][i]]
                        if len(nt_muts) >1:
                            name = [''.join((wt[i],str(i+num_adjust),mut[i])) for i in inds]
                            name = ', '.join(name)
                            mut_type = 'Silent'
                    elif len(inds) >1:
                        name = [''.join((wt[i],str(i+num_adjust),mut[i])) for i in inds]
                        name = ', '.join(name)
                        mut_type = 'Silent'
                except:
                    name = 'Unknown'
                    
        elif len(wt_aa) - len(allele_aa) == 1:
            mut_pos = [i for i in range(len(allele_aa)) if wt_aa[i] != allele_aa[i]]
            if len(mut_pos) > 0:
                first_mut = mut_pos[0]
                allele_aa = allele_aa[:first_mut] + '-' + allele_aa[first_mut:]
            else:
                allele_aa = allele_aa + '-'
            mut_pos = [i for i in range(len(allele_aa)) if wt_aa[i] != allele_aa[i]]
            if len(mut_pos) == 1:
                mut_pos = mut_pos[0]
                mut_type = 'Single Deletion'
                name = wt_aa[mut_pos] + str(mut_pos+num_adjust) + 'del'

        elif '_' in allele_aa:
            shorter = min([len(allele_aa),len(wt_aa)])
            mut_pos = [i for i in range(shorter) if wt_aa[i] != allele_aa[i]]
            stop_site = allele_aa.find('_')
            if len(mut_pos) == 1 and stop_site == mut_pos[0]:
                mut_pos = mut_pos[0]
                mut_type = 'Precise Stop'
                name = wt_aa[mut_pos] + str(mut_pos+num_adjust) + '*'
            elif len(mut_pos) > 1:
                mut_type = 'Frame Shift'
                name = wt_aa[mut_pos[0]] + str(mut_pos[0]+num_adjust) + 'fs'
    return (name,mut_type)

def lesion_checker(al, x, cutsites): #checks to see if deletions are real (extend from the cutsite)
    rels = x.split("_")
    if rels[0] == "HDR":
        return True
    if rels[0] == "HDRp":
        return True
    if len(rels) == 1 and rels[0] == "":
        return True

    #hyphen positions
    hp = [i for i, val in enumerate(al) if al[i] == "-"]
    new_list = cutsites.copy()
    for cut in cutsites:
        rel_hp = len([i for i in hp if i < cut])
        while rel_hp>0:
            new_list.append(cut-rel_hp)
            rel_hp = rel_hp -1    
    real = [r for r in rels if int(r) in new_list]
    if len(real) == 0:
        return False
    else:
        return True

def dpfilter(x, al, ref): #filters out any partial HDR reads or deletion reads with less than two nt changed
    if x == "HDR":
        return True
    if x == "HDRp":
        mismatch = [i for i, val in enumerate(al) if val != ref[i]]
        if len(mismatch) <2:
            return False
        else:
            return True
    if "_" in x:
        return True
    else:
        if len(x) == 0:
            return True
        else:
            return False

def sub_filter(x, y, al, ref): #filters out any partial HDR substitutions with leass than 2 nt changed or substitutions not associated with HDR or not next to a deletion
    if len(y) == 0:
        return True
    if y == "HDR":
        return True
    if y == "HDRp":
        mismatch = [i for i, val in enumerate(al) if val != ref[i]]
        if len(mismatch) <2:
            return False
        else:
            return True

    return False  
################### Change Trimming parameters per gene ###################
def parse_files(folder,amp_legend_df,output_directory_path,oldpath,amp,trim_from_left,trim_from_right,insertions,amp_seq,amp_exon_seq,protdnaseq,amp_aa,amp_guides_df,rel_HDR):
    reads_df = pd.DataFrame()

    trimming_dict = {
    '1': (40, -40),
    '2': (60, -20),
    '3': (20, -40),
    '4': (5, -1),
    '5': (1, -40),
    '6': (5, -1*(234-188)),
    '7': (25,-1),
    '7a': (70,-5),
    '7b': (5,-30),
    '8': (30, -20),
    '9': (40, -26),
    '10': (35, -10),
    '11': (37, -20),
    '12': (18, -40),
    '13': (100, -10),
    '14': (34, -1*(210-140)),
    '15': (90, -1),
    '16': (60, -1*(232-198)),
    }

    file_name = amp_legend_df[amp_legend_df["file_path"] == folder]["output_file_name"].to_list()[0]
    #make path for individual file
    subpath = os.path.join(output_directory_path,file_name)
    if os.path.isdir(subpath) == False:
        os.mkdir(subpath)

    #check to see if the CRISPResso file exists and is complete
    crispresso_dir = os.path.join(oldpath,folder)
    zip_file = os.path.join(crispresso_dir,'Alleles_frequency_table.zip')

    if os.path.exists(zip_file):
        print(folder)
        with ZipFile(zip_file, 'r') as zf:
            zf.extractall(crispresso_dir)
            parse_options = csv.ParseOptions(delimiter="\t")
            reads_df = csv.read_csv(os.path.join(crispresso_dir,'Alleles_frequency_table.txt'),parse_options=parse_options).to_pandas()
            
            reads_df = reads_df.drop(["n_deleted","Read_Status","n_inserted","n_mutated","%Reads","Reference_Name"], axis = 1)
            reads_df = reads_df.rename(columns={("#Reads"):(file_name+"_#Reads"), "Aligned_Sequence":"Aligned_Sequence_untrimmed", "Reference_Sequence":"Reference_Sequence_untrimmed"}) 
            # trim reads
            ############################### Can add additional trimming parameters here if you have amplicon-specific trimming you want to do##############################################################
            
            if amp in trimming_dict.keys():
                left,right = trimming_dict[amp]
            else:
                left = 3
                right = -3
            
            reads_df["Aligned_Sequence"] = reads_df["Aligned_Sequence_untrimmed"].str[trim_from_left+left:trim_from_right+right]
            reads_df["Reference_Sequence"] = reads_df["Reference_Sequence_untrimmed"].str[trim_from_left+left:trim_from_right+right]    
            reads_df = reads_df.drop(['Aligned_Sequence_untrimmed', 'Reference_Sequence_untrimmed'], axis =1)

            dlst = [] # record of amplicons that are dropped from the analysis
            elst = [{"Category":"Starting_number_total_reads","Num_unique": len(reads_df.index), "num_reads": reads_df[(file_name+"_#Reads")].sum()}] #record of amplicons that are dropped from the analysis
            
            #Gets rid of insertions 
            if insertions == False :
                no = reads_df[reads_df["Reference_Sequence"].str.count("-") != 0]
                dlst.append(("insertions",no))
                elst.append({'Category': "insertions","Num_unique": len(no.index),  "num_reads": no[[i for i in no.columns if "Reads" in i]].sum().sum()})
                reads_df = reads_df[reads_df["Reference_Sequence"].str.count("-") == 0]

            if len(reads_df.index) < 1: #if getting rid of insertions makes the dataframe empty, skip the rest of the analysis and move on to next TR
                return pd.DataFrame()

            #check WT amplicon and AA sequence to make sure there are no issues
            tempdf = reads_df.copy()
            al = tempdf[tempdf["Aligned_Sequence"] == tempdf["Reference_Sequence"]]["Reference_Sequence"].to_list()
            if len(al) > 0:
                al = al[0]
            else:
                print('Error identifying WT sequence in the Alleles Frequency table. Perhaps no WT Allele present')
                if len(tempdf[tempdf['Reference_Sequence'].apply(lambda x: x in amp_seq)]['Reference_Sequence'].tolist()) >0:
                    al = tempdf[tempdf['Reference_Sequence'].apply(lambda x: x in amp_seq)]['Reference_Sequence'].tolist()[0]
                else:
                    print('Reference sequence not found in amplicon sequence. Check trimming parameters!')
                    return pd.DataFrame()
                #return pd.DataFrame()
            del tempdf

            if al not in amp_seq:
                print('WT sequence not found in amplicon sequence. Check trimming parameters!')
                return pd.DataFrame()
            if amp_exon_seq not in al:
                print('Amplicon exon sequence not found in sequencing alignment! Check trimming parameters')
                return pd.DataFrame()
            if amp_exon_seq not in protdnaseq:
                print('Exon sequence does not match provided cDNA')
                return pd.DataFrame()
            exon_loc = protdnaseq.find(amp_exon_seq)
            front_bases = 0
            end_bases = ''
            if exon_loc%3 !=0:
                front_bases = 3 - exon_loc%3
                amp_exon_seq = amp_exon_seq[front_bases:]
                exon_loc = protdnaseq.find(amp_exon_seq)
            if len(amp_exon_seq)%3 != 0:
                end_bases = protdnaseq[exon_loc+len(amp_exon_seq):exon_loc+len(amp_exon_seq)+(3-len((amp_exon_seq))%3)]
            if translate((amp_exon_seq+end_bases)) != amp_aa:
                print('AA sequence provided does not match translated exon sequence')
                print('Provided sequence: %s' % amp_aa)
                print('Translated sequence: %s' % translate((amp_exon_seq+end_bases)))
            
            potential_cutsite_idx = []
            # for wt allele, go through and find where the nt idx left and right of all the relevant sgRNA would be
            amp_guides_df["cutsite_idx"] = amp_guides_df.apply(lambda x: sgFinder(x['sgRNA seq'],al), axis =1)
            for x in amp_guides_df["cutsite_idx"].to_list():
                for i in x:
                    potential_cutsite_idx.append(i)
                    potential_cutsite_idx = unique(potential_cutsite_idx)
            
            reads_df = reads_df.reset_index()
            
            # go through alleles, and first identify potential cut site, then which subs are within 2bp of a potential cut site   
            # ############################# Validate Subs is where it's printing out alleles it doesn't like    #############################          
            reads_df["mismatch_idx"],reads_df["Aligned_Sequence_PCR_Error_fixed"], reads_df['sub_idx'],  reads_df['ins_idx'] , reads_df["del_idx"], reads_df["HDR"] = zip(*(reads_df.apply(lambda x: validate_subs(x["Aligned_Sequence"],x["Reference_Sequence"],rel_HDR,potential_cutsite_idx), axis =1)))
            if len(reads_df.index) < 1:
                return pd.DataFrame()
            df2t = pa.Table.from_pandas(reads_df)
            csv.write_csv(df2t, os.path.join(subpath,"PCRerrorcorrection_mutID_multitag.csv"))   

            #Start filtering out the reads not relevant to the analysis
            #before filter, keep track of what i'm filtering out accordingly
            no = reads_df[reads_df["Aligned_Sequence_PCR_Error_fixed"] == "chimera"]
            dlst.append(("chimera",no))
            elst.append({'Category': "chimera","Num_unique": len(no.index),  "num_reads": no[[i for i in no.columns if "Reads" in i]].sum().sum()})
            no = reads_df[reads_df["Aligned_Sequence_PCR_Error_fixed"] == "Alignment_Issue"]
            dlst.append(("Alignment_Issue",no))
            elst.append({'Category': "Alignment_Issue","Num_unique": len(no.index),  "num_reads": no[[i for i in no.columns if "Reads" in i]].sum().sum()})
            no = reads_df[reads_df["Aligned_Sequence_PCR_Error_fixed"] == "Multiple possible"]
            dlst.append(("Multiple possible",no))
            elst.append({'Category': "Multiple possible","Num_unique": len(no.index),  "num_reads": no[[i for i in no.columns if "Reads" in i]].sum().sum()})
            
            
            #filter
            reads_df = reads_df[reads_df["Aligned_Sequence_PCR_Error_fixed"].str.len() > 40]
            if len(reads_df.index) < 1: #if getting rid of very short sequences makes the dataframe empty, skip this part of the loop and move on to the next TR
                return pd.DataFrame()  

            # This next filter will get rid of any non-HDR deletions which don't extend from the cutsite. This is maybe be vunerable to " (cutsite)A---A" to "(cutsite)---AA" alignment noise
            # group into consecutiveish (within 2bp) stretches to consider as one gene editing event
            reads_df["lesion_checker"] = reads_df.apply(lambda x: lesion_checker(x["Aligned_Sequence"],x["del_idx"],potential_cutsite_idx), axis =1)
            
            no = reads_df[reads_df["lesion_checker"]==False]
            dlst.append(("lesion_checker",no))
            elst.append({'Category': "lesion_checker","Num_unique": len(no.index),  "num_reads": no[[i for i in no.columns if "Reads" in i]].sum().sum()})
            
            reads_df = reads_df[reads_df["lesion_checker"]==True]
            reads_df = reads_df.drop(["lesion_checker"], axis =1)

            #getting rid of single nucleotide deletions, substitution events that aren't HDR, or partial HDR events resulting in <2 mismatches
            reads_df["del_counter"] = reads_df.apply(lambda x: dpfilter(x["del_idx"],  x["Aligned_Sequence_PCR_Error_fixed"], x["Reference_Sequence"]), axis =1)
            
            no = reads_df[reads_df["del_counter"] == False]
            dlst.append(("del_counter",no))
            elst.append({'Category': "del_counter","Num_unique": len(no.index),  "num_reads": no[[i for i in no.columns if "Reads" in i]].sum().sum()})
            
            reads_df = reads_df[ (reads_df["del_counter"] ==True)]
            
            #subs
            reads_df["sub_filter"] = reads_df.apply(lambda x: sub_filter(x["del_idx"], x["sub_idx"], x["Aligned_Sequence_PCR_Error_fixed"], x["Reference_Sequence"]), axis =1)
            
            no = reads_df[reads_df["sub_filter"] == False]
            dlst.append(("sub_filter",no))
            elst.append({'Category': "sub_filter","Num_unique": len(no.index),  "num_reads": no[[i for i in no.columns if "Reads" in i]].sum().sum()})

            reads_df = reads_df[reads_df["sub_filter"] == True]
            
            reads_df = reads_df.drop(["del_counter", "sub_filter"], axis =1)
            if len(reads_df.index) < 1: #if dropping these reads makes the dataframe empty, exit and proceed to the next file
                return pd.DataFrame()
            
            #filtering complete, compile
            elst.append({'Category': "post_filtering_actual_file","Num_unique": len(reads_df.index),  "num_reads": reads_df[[i for i in no.columns if "Reads" in i]].sum().sum()})
            df3 = pd.concat([i for k,i in dlst], keys = [k for k,i in dlst]).reset_index()
            df4 = pd.DataFrame(elst)
            df4.to_csv(os.path.join(subpath,"summary_of_PCR_error.csv"))
            df3t = pa.Table.from_pandas(df3)
            csv.write_csv(df3t, os.path.join(subpath,"summary_of_PCR_error_reads_file.csv" ))

            return reads_df
                 
    else:
        print('Warning: %s could not be analyzed. Check CRISPResso outputs' %folder)
        return reads_df

def end_splice_site(start_pos, splice_nt, al): #identify where the end of the splice site is, if possible and intact
    if start_pos == -1:
        return -1
    else:
        num = al.count(splice_nt)
        if num >=2:
            tentative = al.rfind(splice_nt)
            return tentative

        tentative = al.find(splice_nt)
        if tentative > start_pos:
            return tentative
        elif tentative == -1:
            return -1
        else:
            tentative = al.find(splice_nt, tentative+len(splice_nt))
            if tentative == -1:
                return -1
            elif tentative > start_pos:
                return tentative
            else: 
                tentative = al.find(splice_nt, tentative + len(splice_nt))
                if tentative == -1:
                    return -1
                elif tentative > start_pos:
                    return tentative
                else:
                    print(al)
                    print(start_pos)
                    print(splice_nt)
                    print("error")
                    exit      

def translated_contextually(start_splice, end_splice, al, splice_start_status, splice_end_status, frame,last2ntbefore,first2ntnext, hdr, ref, with_hyphen): #translate sequence within the context of the exon 
    if hdr == True:
        splice_start_status = 0
        splice_end_status = 0
    if splice_start_status == -1:
        return "False"
    if splice_end_status == -1:
        return "False"
    if hdr != True: 
        start_from_here = al.find(start_splice)
        end_here = al.rfind(end_splice)
    else:
        if "---" in with_hyphen:
            start_from_here = ref.find(start_splice)
            end_here = ref.rfind(end_splice)-3
        else:
            start_from_here = ref.find(start_splice)
            end_here = ref.rfind(end_splice)
    
    if (len(end_splice) > 3) and (len(start_splice) >3):
        s = al[start_from_here+3:end_here+3]
        
    else:
        print("error in defining start and end of exon to define wt allele") 
        return "False"

    #get preceeding nucleotides to fix codons
    if frame != 0 and len(last2ntbefore)>0:
        s = last2ntbefore[-frame:] + s

    if (len(first2ntnext) >0):
        if len(s) % 3 == 0:
            rel_seq = s
        elif len(s) % 3 == 1:
            rel_seq = s+first2ntnext
        elif len(s) % 3 == 2:
            rel_seq = s+first2ntnext[0]
    else:
        if len(s) % 3 == 0:
            rel_seq = s
        elif len(s) % 3 == 1:
            rel_seq = s[:-1]
        elif len(s) % 3 == 2:
            rel_seq = s[:-2]

    rel_prot = translate(rel_seq)
    if "_" in rel_prot:
        rel_prot = rel_prot[:rel_prot.find("_")+1]   

    if len(rel_prot) > 0:
        return rel_prot
    else:
        return "Unclear"

def get_genotypes(folder,folderpath,crispresso_path):
    if os.path.isfile(os.path.join(folderpath,folder,'memorysaver_collapsed.csv')):
        memsaver_df = pd.read_csv(os.path.join(folderpath,folder,'memorysaver_collapsed.csv'))
        #check for NaN
        nan_dex = memsaver_df[memsaver_df.isna().any(axis=1)].index.tolist()
        if len(nan_dex)>0:
            if 'WT' in memsaver_df['mutation_name'].tolist():
                wt = memsaver_df[memsaver_df['mutation_name'] == 'WT'].translated_where_possible.tolist()[0]
                for uk in nan_dex:
                    if memsaver_df.iloc[uk,0] == wt:
                        memsaver_df.iloc[uk,1] = 'Unknown'
                        memsaver_df.iloc[uk,2] = 'WT PCR-Corrected'
                    else:
                        memsaver_df.iloc[uk,1] = 'Unknown'
                        memsaver_df.iloc[uk,2] = 'Unknown'
        cols = [x for x in memsaver_df.keys().tolist() if 'Reads' not in x]
        #find out how many reads aren't classified by our metric
        crisp_dir = [x for x in os.listdir(crispresso_path) if folder in x and os.path.isdir(os.path.join(crispresso_path,x))]
        if len(crisp_dir) == 1:
            memsaver_df = memsaver_df.groupby(cols).sum().reset_index()
            crisp_dir = os.path.join(crispresso_path,crisp_dir[0])
            reads_dir = pd.read_csv(os.path.join(crisp_dir,'CRISPResso_mapping_statistics.txt'),delimiter='\t')
            num_total = reads_dir['READS ALIGNED'].tolist()[0]
            unassigned = num_total - memsaver_df[folder+'_#Reads'].sum()
            tdf = pd.DataFrame(columns = ['translated_where_possible', 'mutation_type', 'mutation_name', folder+'_#Reads', '%Reads', 'pct_sum'],
                               data=[['','Unclassified','Unclassified',int(unassigned),int(unassigned)/num_total,0]])
            tdf.astype({folder+'_#Reads':'int32'}).dtypes
            memsaver_df = pd.concat([memsaver_df,tdf])
            memsaver_df = memsaver_df.sort_values(by=folder+'_#Reads',ascending=False)
            memsaver_df.to_csv(os.path.join(folderpath,folder,'memorysaver_collapsed.csv'),index=False)
            return[memsaver_df['mutation_name'].tolist(),
                   memsaver_df[folder+'_#Reads'].tolist(),
                   memsaver_df['translated_where_possible'].tolist(),
                   [x/num_total for x in memsaver_df[folder+'_#Reads'].tolist()],
                   memsaver_df['mutation_type'].tolist(),
                   num_total]
        else:
            print('could not find read file')
            return [[],[],[],[],[],0]
    else:
        return [[],[],[],[],[],0]

def expand_df(row):
    temp_df = pd.DataFrame()
    temp_df['mutation_names'] = row.mutation_names
    temp_df['mutation_readcounts'] = row.mutation_readcounts
    temp_df['well'] = row.well
    return temp_df

def inds_to_keep(df,prefix,pc,rc):
    inds = []
    for i,r in df.iterrows():
        if i==0:
            inds = inds+[True]
        elif df.iloc[i-1]['pct_sum'] <=pc and r[prefix+'_#Reads']>=rc:
            inds = inds+[True]
        else:
            inds = inds+[False]
    return inds

# Global settings
trim_from_left = 2
trim_from_right = -2
insertions = True # do you want to annotate insertions? 
DEBUG = True
HDR = True # do you have HDR templates that you would like to analyze?

################################ These are constant per gene ################################
# Set paths for input/output files and gene information
# Input Gene information
with open(r"/data/annika/96WellGenotyping/2024-03-21_POT1Array_4plates/CommonInputs/POT1.txt" , "r") as file:
    gene_seq = file.read().upper()
with open(r"/data/annika/96WellGenotyping/2024-03-21_POT1Array_4plates/CommonInputs/POT1_Prot.txt", "r") as file:
    prot_seq = file.read().upper()
with open(r"/data/annika/96WellGenotyping/2024-03-21_POT1Array_4plates/CommonInputs/POT1_dna_seq_just_exons.txt", "r") as file:
    protdnaseq = file.read().upper()

guide_ref = pd.read_excel('/data/annika/96WellGenotyping/2024-03-21_POT1Array_4plates/CommonInputs/POT1_sgRNA_Seq.xlsx')

################################ These need to change per plate run ################################
exon_seqs_df = pd.read_excel('/data/annika/96WellGenotyping/2024-03-21_POT1Array_4plates/Amp3_sg18_RB_Ala/Input/A3g13Amplicon.xlsx')
exon_seqs_df['Amplicon #'] = exon_seqs_df['Amplicon #'].apply(lambda x: str(x))
amp = '3'
guide = 18
#Input path information
oldpath = r"/data/annika/96WellGenotyping/2024-03-21_POT1Array_4plates/Amp3_sg18_RB_Ala/CRISPRessoBatch_on_POT1_amp3_sg18_RB_ALA"
#Output path information (make directory if needed)
dpath = r"/data/annika/96WellGenotyping/2024-03-21_POT1Array_4plates/Amp3_sg18_RB_Ala/Output"
if os.path.exists(dpath) == False:
    os.mkdir(dpath)
################################################################################################################################

guide_ref = guide_ref[guide_ref['sgRNA # oPool Name']==guide].copy()

if HDR:    
    HDR_lib_design = pd.read_excel(r"/data/annika/96WellGenotyping/2024-03-21_POT1Array_4plates/CommonInputs/RB_Libraries.xlsx")
    HDR_lib_design = HDR_lib_design[HDR_lib_design['sgRNA # oPool Name']==guide].copy()
    HDR_lib_design = HDR_lib_design.reset_index(drop=True)
    
    sgRNA_list = HDR_lib_design[['sgRNA # oPool Name',  'Exon #', 'sgRNA seq', "region around sgRNA that is exon"]].drop_duplicates(keep='first').reset_index(drop=True)
    sgRNA_list = sgRNA_list[sgRNA_list['sgRNA # oPool Name']==guide].copy()
    #Add amplicon information to sgRNA list
    sgRNA_list['Amplicon'] = [str(guide_ref[guide_ref['sgRNA sequence'] == guide]['Amplicon'].tolist()[0]) for guide in sgRNA_list['sgRNA seq']]
    #Drop any problematc templates where tagging didn't work or the length of the region of interest was changed (insertions,etc)
    HDR_lib_design = HDR_lib_design[HDR_lib_design['region of interest prechange'].str.len() == HDR_lib_design["changed region of interest"].str.len() ]
    HDR_lib_design = HDR_lib_design[HDR_lib_design['with_tagging_mutations_for_single_muts'].str.len() >1]
    HDR_lib_design["mini_seqs"] = [HDR_lib_design.iloc[i]['with_tagging_mutations_for_single_muts'][HDR_lib_design.iloc[i]['HDR window'].find(HDR_lib_design.iloc[i]['region of interest prechange']):HDR_lib_design.iloc[i]['HDR window'].find(HDR_lib_design.iloc[i]['region of interest prechange'])+len(HDR_lib_design.iloc[i]['region of interest prechange'])] for i in HDR_lib_design.index.tolist()]

else:
    sgRNA_list = guide_ref.copy()
    sgRNA_list = interpret_guides(gene_seq,exon_seqs_df,sgRNA_list)

#for the amplicon, extract relevant information
####################### May need to change this if the CRISPResso Output file name structure changes #######################
amp_legend_df = pd.DataFrame()    
amp_legend_df['file_path'] = [x for x in os.listdir(oldpath) if os.path.isdir(os.path.join(oldpath,x))]
amp_legend_df['well'] = amp_legend_df.file_path.apply(lambda x : x[-3:])
amp_legend_df['column'] = amp_legend_df["well"].apply(lambda x: x[-2:])
amp_legend_df['row'] = amp_legend_df["well"].apply(lambda x: x[0])
amp_legend_df['barcode'] = amp_legend_df["file_path"].apply(lambda x: re.split('(\d+)',x.split('_')[2])[2][:-1])
amp_legend_df['output_file_name'] = amp_legend_df.well
amp_legend_df['amplicon'] = amp
amp_seq = exon_seqs_df.loc[exon_seqs_df['Amplicon #'] == amp]['Amplicon Sequence'].item().upper()
amp_legend_df['amplicon_seq'] = amp_seq

amp_aa = exon_seqs_df.loc[exon_seqs_df['Amplicon #'] == amp]['Protein Sequence'].item()
amp_exon_seq = exon_seqs_df.loc[exon_seqs_df['Amplicon #']==amp]['Exon Sequence'].item()

#find position of exon within the amplicon and identify splice sites
amp_legend_df.loc[:,"Exon_Sequence"] = amp_exon_seq
exon_start = amp_seq.find(amp_exon_seq)
start_splice = amp_seq[exon_start-3:exon_start+3]
amp_legend_df.loc[:, "start_splice"] = start_splice
exon_end = exon_start + len(amp_exon_seq)
end_splice = amp_seq[exon_end-3:exon_end+3]
amp_legend_df.loc[:,"end_splice"] = end_splice

# get relevant sgRNA seqs
amp_guides_df = sgRNA_list.copy()
amp_guides_df = amp_guides_df[amp_guides_df['Amplicon'] == amp]

if HDR: #if an HDR file is provided, identify all relevant HDR and partial HDR templates for this specific amplicon
    rel_HDR = identify_relevant_HDR(HDR_lib_design,amp_guides_df,amp_seq,protdnaseq)


# Make subdirectory for amplicon information
Title = str(amp)
output_directory_path =  dpath
if os.path.isdir(output_directory_path) == False:
    os.mkdir(output_directory_path)
rel_HDR.to_csv(os.path.join(output_directory_path,'relevantHDRtemplates.csv'))

#for the relevant amplicon, start cycling through each CRISPResso folder and annotating reads
print("Extracting info from Amplicon #%s CRISPResso Folders..." % amp)
Crispresso_output_folder_list = amp_legend_df["file_path"].to_list()
for folder in Crispresso_output_folder_list:
    #classify and filter the reads
    try:
        reads_df = parse_files(folder,amp_legend_df,output_directory_path,oldpath,amp,trim_from_left,trim_from_right,insertions,amp_seq,amp_exon_seq,protdnaseq,amp_aa,amp_guides_df,rel_HDR)
        if len(reads_df.index) > 1:
            #find if splicing junctions are intact , and translate when possible  
            reads_df["Aligned_Sequence_PCR_Error_fixed_no_hyphens"] = reads_df["Aligned_Sequence_PCR_Error_fixed"].str.replace("-","")
            if len(start_splice) > 3:
                reads_df['intact_splice_start'] = reads_df["Aligned_Sequence_PCR_Error_fixed_no_hyphens"].str.find(start_splice)
            else:
                reads_df['intact_splice_start'] = 0
            
            if len(end_splice) > 3:
                reads_df['intact_splice_end'] = reads_df.apply(lambda x: end_splice_site(x["intact_splice_start"], end_splice, x["Aligned_Sequence_PCR_Error_fixed_no_hyphens"]), axis =1)
            else:
                reads_df["intact_splice_end"] = 0
            
            #find the frame for translation and get nts from neighboring exons to fix partial codon issues
            frame = protdnaseq.find(amp_exon_seq)%3
            last2nt_before = ''
            first2nt_next = ''

            if frame != 0 and protdnaseq.find(amp_exon_seq) !=0:
                last2nt_before = protdnaseq[protdnaseq.find(amp_exon_seq)-2:protdnaseq.find(amp_exon_seq)]
            if protdnaseq.find(amp_exon_seq) + len(amp_exon_seq) < len(protdnaseq)-1:
                first2nt_next = protdnaseq[protdnaseq.find(amp_exon_seq)+len(amp_exon_seq):protdnaseq.find(amp_exon_seq)+len(amp_exon_seq)+2]
            reads_df["translated_where_possible"] = reads_df.apply(lambda x: translated_contextually(start_splice, end_splice, x["Aligned_Sequence_PCR_Error_fixed_no_hyphens"], x["intact_splice_start"], x["intact_splice_end"], frame,last2nt_before,first2nt_next, x["HDR"],x["Reference_Sequence"], x["Aligned_Sequence_PCR_Error_fixed"]), axis =1)                          
            #reads_df['splicing_intact'] = reads_df.apply(splicing_intact,axis=1)

            if len(unique(reads_df[reads_df['mismatch_idx'] == 'WT']['translated_where_possible'].tolist())) == 1:
                wt_aa = unique(reads_df[reads_df['mismatch_idx'] == 'WT']['translated_where_possible'].tolist())[0]
                if wt_aa != amp_aa:
                    print('Check input AA sequence')
                    print('Input AA: %s' % amp_aa)
                    print('Translated AA: %s' % wt_aa)
            
            else: 
                print('Issue classifying WT allele; no allele classification attempted at this stage.')        
                
            mut_name,mut_type = zip(*reads_df.apply(lambda x: attempt_name(wt_aa, prot_seq, x,amp_exon_seq,protdnaseq,amp), axis=1))
            reads_df['mutation_type'] = mut_type
            reads_df['mutation_name'] = mut_name


            file_name = amp_legend_df[amp_legend_df["file_path"] == folder]["output_file_name"].to_list()[0]
            #make path for individual file
            subpath = os.path.join(output_directory_path,file_name)

            df2t = pa.Table.from_pandas(reads_df)
            
            csv.write_csv(df2t, os.path.join(subpath,"PCRerrorcorrection_mutID_multitag_splice_translate.csv") )
            memsaver_df = reads_df.copy()
            memsaver_df = memsaver_df.drop(['Reference_Sequence',
                    'mismatch_idx', 'Aligned_Sequence_PCR_Error_fixed_no_hyphens', 'sub_idx',
                    'ins_idx', 'del_idx', 'intact_splice_start',
                    'intact_splice_end', "HDR",'index'], axis = 1)
            memsaver_df.to_csv(os.path.join(subpath, "memorysaver.csv"), index=False)

            memsaver_df = memsaver_df.drop(["Aligned_Sequence_PCR_Error_fixed", "Aligned_Sequence"],axis =1)
            cols = [i for i in memsaver_df.columns if "Reads" not in i]
            memsaver_df = memsaver_df.groupby(cols).sum().reset_index()
########################################### After collapsing alleles, DISCARD ALLELES WITH JUST A SINGLE READ OR LOWLY-REPRESENTED ALLELES (BOTTOM 2%) ###########################################
            memsaver_df = memsaver_df.sort_values(by=file_name+'_#Reads',ascending=False).reset_index(drop=True)
            total_reads = pd.read_csv(os.path.join(oldpath,folder,'CRISPResso_mapping_statistics.txt'),delimiter='\t')['READS ALIGNED'].tolist()[0]
            memsaver_df['%Reads'] = memsaver_df[file_name+'_#Reads'].apply(lambda x: x/total_reads)
            memsaver_df['pct_sum'] = memsaver_df['%Reads'].cumsum()
            pc = .98 #fraction of reads to keep (ex. if .98, the bottom 2% of all alleles are dropped)
            rc = 2 #minimum read count (ex. if 2, any alleles with less than 2 reads are dropped)
            keep = inds_to_keep(memsaver_df,file_name,pc,rc)
            memsaver_df = memsaver_df[keep].copy()
            memsaver_df.to_csv(os.path.join(subpath, "memorysaver_collapsed.csv"), index=False)
    except:
        print('%s Warning: FOLDER FAILED' % folder)

axis_row_dict = {
    'A':0,
    'B':1,
    'C':2,
    'D':3,
    'E':4,
    'F':5,
    'G':6,
    'H':7
}

amp_legend_df = amp_legend_df.drop(['start_splice','end_splice'],axis=1)
amp_legend_df['temp'] = amp_legend_df['output_file_name'].apply(lambda x: get_genotypes(x,output_directory_path,oldpath))
amp_legend_df['mutation_names'] = amp_legend_df['temp'].apply(lambda x: x[0])
amp_legend_df['mutation_readcounts'] = amp_legend_df['temp'].apply(lambda x: x[1])
amp_legend_df['translations'] = amp_legend_df['temp'].apply(lambda x: x[2])
amp_legend_df['mutation_pct'] = amp_legend_df['temp'].apply(lambda x: x[3])
amp_legend_df['mutation_types'] = amp_legend_df['temp'].apply(lambda x: x[4])
amp_legend_df['total_reads'] = amp_legend_df['temp'].apply(lambda x: x[5])

amp_legend_df = amp_legend_df.drop(['temp','barcode','output_file_name','amplicon_seq','Exon_Sequence'],axis=1)
amp_legend_df = amp_legend_df.sort_values(by="well").reset_index(drop=True)


amp_legend_df.to_csv(os.path.join(dpath,'Amp'+amp+"PerWellGenotypingSummary.csv"),index=False)

#generate allele matrix
allele_matrix = pd.DataFrame()
for index,row in amp_legend_df.iterrows():
    tdf = pd.DataFrame()
    tdf['mutation_name'] = row['mutation_names']
    tdf['mutation_type'] = row['mutation_types']
    tdf['translation'] = row['translations']
    tdf[row['well']] = row['mutation_pct']
    tdf = tdf.set_index(['mutation_name','mutation_type','translation'])
    allele_matrix = pd.concat([allele_matrix,tdf],axis=1)

allele_matrix = allele_matrix.reset_index()
wt = allele_matrix[allele_matrix['mutation_name']=='WT']['translation'].tolist()[0]
allele_matrix['mutation_name'] = allele_matrix.apply(
    lambda x: 'WT PCR-Corrected' if x['mutation_type']=='Unknown' and x['translation']==wt else x['mutation_name'],axis=1)
allele_matrix = allele_matrix.groupby(by=['mutation_name','mutation_type','translation']).sum().reset_index()

######################### Make Figure #########################
all_rows = ['A','B','C','D','E','F','G','H']
all_columns = ['01','02','03','04','05','06','07','08','09','10','11','12']
all_wells = [x+y for x in all_rows for y in all_columns]

left_outs = [x for x in all_wells if x not in allele_matrix.columns.tolist()]
left_outs

for well in left_outs:
    allele_matrix[well] = 0

allele_matrix.to_csv(os.path.join(dpath,'Amp'+amp+'AlleleMatrix.csv'),index=False)

#check all wellsums
if len(allele_matrix.columns)-3 != 96:
    print('Caution: %d wells appear to be missing from the allele matrix' % (96-(len(allele_matrix.columns)-3)))
if len([x for x in allele_matrix[all_wells].sum() if round(x,15)!=1]) > 0:
    print('Caution: %d wells do not sum to 1 (to 15 sig digits)' % len([x for x in allele_matrix[all_wells].sum() if round(x,15)!=1]))
    checksum_list = [round(x,15)!=1 for x in allele_matrix[all_wells].sum()]
    well_list = [all_wells[i] for i in range(len(checksum_list)) if checksum_list[i] == True]
    print('Check the following wells: %s' % ', '.join(well_list))
elif len(allele_matrix.columns)-3 == 96 and len([x for x in allele_matrix[all_wells].sum() if round(x,15)!=1]) == 0:
    print('96 wells surveyed and all checksums sum to 1!')

axis_row_dict = {
    'A':0,
    'B':1,
    'C':2,
    'D':3,
    'E':4,
    'F':5,
    'G':6,
    'H':7
}

l = allele_matrix.mutation_name.tolist()
cs=cmaps.pride(np.linspace(0.05,0.95,len(l)))

fig,axes = plt.subplots(figsize=(12,8),nrows = 8,ncols=12)
leg_patches = []
maxxsize = max(amp_legend_df.total_reads.tolist())
for well in all_wells:
    if well in amp_legend_df['well'].tolist():
        r = amp_legend_df[amp_legend_df['well'] == well]['total_reads'].item()/maxxsize
        ax0 = axis_row_dict[well[0]]
        ax1 = int(well[1:])-1
        if r>0:
            patches,texts = axes[ax0,ax1].pie(allele_matrix[well].tolist(),colors = cs,radius=1.5*r)
            leg_patches = patches
        else:
            patches,texts = axes[ax0,ax1].pie([])
    else:
        patches,texts = axes[ax0,ax1].pie([])
cols = ['01','02','03','04','05','06','07','08','09','10','11','12']
rows = ['A','B','C','D','E','F','G','H']
for ax, col in zip(axes[0], cols):
    ax.set_title(col)
nlcols = round(np.sqrt(len(leg_patches))/4,0)
for ax, row in zip(axes[:,0], rows):
    ax.set_ylabel(row, rotation=0, size='large')
fig.suptitle('96-well Plate Layout',fontsize=22)
fig.legend(leg_patches,l,loc='center left',bbox_to_anchor=(0.9, 0.5),ncols=nlcols)
fig.text(0.95,0.15,'Maximum Reads: %d' % maxxsize)
fig.savefig(os.path.join(dpath,'Amp'+amp+'GenotypeSummary_ReadCountScaled.png'),bbox_inches='tight')
plt.close()

fig,axes = plt.subplots(figsize=(12,8),nrows = 8,ncols=12)
leg_patches = []
maxxsize = max(amp_legend_df.total_reads.tolist())
for well in all_wells:
    if well in amp_legend_df['well'].tolist():
        r = amp_legend_df[amp_legend_df['well'] == well]['total_reads'].item()/maxxsize
        ax0 = axis_row_dict[well[0]]
        ax1 = int(well[1:])-1
        if r>0:
            patches,texts = axes[ax0,ax1].pie(allele_matrix[well].tolist(),colors = cs,radius=1.5*r)
            leg_patches = patches
        else:
            patches,texts = axes[ax0,ax1].pie([])
    else:
        patches,texts = axes[ax0,ax1].pie([])
cols = ['01','02','03','04','05','06','07','08','09','10','11','12']
rows = ['A','B','C','D','E','F','G','H']
for ax, col in zip(axes[0], cols):
    ax.set_title(col)
nlcols = round(np.sqrt(len(leg_patches))/4,0)
for ax, row in zip(axes[:,0], rows):
    ax.set_ylabel(row, rotation=0, size='large')
fig.suptitle('96-well Plate Layout',fontsize=22)
fig.legend(leg_patches,l,loc='center left',bbox_to_anchor=(0.9, 0.5),ncols=nlcols)
fig.savefig(os.path.join(dpath,'Amp'+amp+'GenotypeSummary.png'),bbox_inches='tight')
plt.close()