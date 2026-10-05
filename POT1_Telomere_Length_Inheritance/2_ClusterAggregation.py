import pandas as pd
import numpy as np
import os
import Levenshtein as L
from pathlib import Path
from glob import glob
from Bio import Align
from inputimeout import inputimeout, TimeoutOccurred
from datetime import datetime
from tqdm import tqdm


class Read_Cluster:
    def __init__(self,key,data,lev_ratio,ind,chrom):
        self.key = key #defines the dataset assoicated with the cluster
        self.data = data #defines the data associated with the cluster in the form of a dictionary
        self.lev_ratio = lev_ratio #defines the levenshtein distance between the cluster tvr and 
        self.ind = ind #defines the index of the cluster in the original data frame
        self.chrom = chrom #defines the chromosome of the cluster

def unique(list1):
    '''return only the unique elements of a list (no duplicates)'''
    x = np.array(list1)
    unique_list = np.unique(x).tolist()
    return unique_list

def align_tvrs_correct_endgaps(tvr1,tvr2):
    # check to see if there is a long stretch of canonical sequence (>50% of TVR length) or high C content that is intervening in either 
    # tvr1 or tvr2 which may complicate analysis. If so, trim to the chromosome end and exclude the terminal portion with high repeats.
    if 'C'*round(len(tvr1)/2) in tvr1:
        repeat_ind = tvr1.find('C'*round(len(tvr1)/2))
        if repeat_ind > 100:
            tvr1 = tvr1[:repeat_ind]
    if 'C'*round(len(tvr2)/2) in tvr2:
        repeat_ind = tvr2.find('C'*round(len(tvr2)/2))
        if repeat_ind > 100:
            tvr2 = tvr2[:repeat_ind]

    aligner = Align.PairwiseAligner()
    #align the two tvrs
    alignment = aligner.align(tvr1, tvr2)[0]

    #determine if there is a stretch of '-' sequence at the beginning or end and replace with 'C' (canonical telomeric sequence)
    gaps = '----------'
    alignment_gaps_replaced = []
    for allele in alignment:
        if allele[-len(gaps):]==gaps:
            for i in range(-1,-len(allele),-1):
                if allele[i]!='-':
                    break
            allele = allele[:i+1]+((-1*i)-1)*'C'
        if allele[0:len(gaps)]==gaps:
            for i in range(len(allele)):
                if allele[i]!='-':
                    break
            allele = i*'C'+ allele[i:]
        alignment_gaps_replaced.append(allele)
    return alignment_gaps_replaced[0],alignment_gaps_replaced[1]

def calc_lev_ratio(input,ref):
    empty = ['nan',np.nan,'NaN',None,'None','',0,'0']
    if input in empty or ref in empty:
        return 0
    if str(input)=='nan' or str(ref)=='nan':
        return 0
    elif input == ref:
        return 1
    try:
        input,ref = align_tvrs_correct_endgaps(input,ref)
        if len(input)!=len(ref):
            print('StopHere!')
        min_length = min([len(input),len(ref)])
        l = L.ratio(ref[:min_length],input[:min_length])
        return l
    except:
        print(f'Error in calculating lev. ratio between the following: \ninput: {input}\nref: {ref}')
        return -1

def match_clusters(row,df_dict,keys_lst,data_cols,thresh):
    clust_list = []
    ref_tvr = row['tvr_consensus']
    for key in keys_lst:
        warnings = ''

        comp_df = df_dict[key].copy()
        if len(comp_df)==0:
            continue
        comp_df['temp'] = comp_df.tvr_consensus.apply(lambda x: calc_lev_ratio(x,ref_tvr))
        if np.nanmax(comp_df.temp) >= thresh:
            tdf = comp_df[comp_df.temp >= thresh].copy()
            if len(tdf)>1:
                warnings = warnings+'multiple_high_similarity,'
                tdf = tdf[tdf.temp == max(tdf.temp)].copy()
            if len(tdf)>1:
                warnings =  warnings+'equivalent_similarity,'
                print('Alleles with equal tvr lev ratios... defaulting to highest read count that is not an intersitital allele')
                tdf = tdf[tdf.allele_id.apply(lambda x: 'i' not in str(x))].copy()
            if len(tdf)>1:
                tdf['temp2'] = tdf.read_lengths.apply(lambda x: len(x.split(',')))
                tdf = tdf[tdf.temp2 == max(tdf.temp2)].copy()
            ind = tdf.index
            new_cols = [f"{key}_{x}" for x in data_cols]
            tdf = tdf.rename(columns=dict(zip(data_cols,new_cols)))
            clust = Read_Cluster(key,
                                 tdf[new_cols].to_dict(orient='list')|{f"{key}_warnings":warnings}|{f"{key}_lev_ratio":tdf.temp.item()},
                                 tdf.temp.item(),
                                 ind,tdf[f"{key}_#chr"].item())
            clust_list.append(clust)
    if len(clust_list)==0:
        print('no matches!')
    return clust_list

def read_and_filter_tsv(path,LEV_THRESHOLD,prefix):
    telogator_tsv = pd.DataFrame()
    ##### Parameters for cluster filtering #####
    p75_thresh = 300 # measured telomere length must be over this value
    tvr_canon_repeats = 'C'*500 # and the tvr must not have a long stretch of canonical repeats
    try:
        print(f"Reading in sample {prefix}...")
        telogator_tsv = pd.read_csv(path,delimiter='\t').reset_index(drop=True)
        starting_len = len(telogator_tsv)
        print(f'{starting_len} starting clusters...')

        #drop any reads that don't have a tvr for cluster comparison
        telogator_tsv = telogator_tsv.dropna(subset='tvr_consensus').reset_index(drop=True)
        print(f"{starting_len-len(telogator_tsv)} clusters dropped based on lack of TVR...")

        #drop any reads that are called as interstital alleles
        interstitial = len(telogator_tsv[telogator_tsv.allele_id.apply(lambda x: 'i' in str(x))])
        if interstitial>0:
            print(f"{interstitial} clusters dropped based on interstitial classification...")
        telogator_tsv = telogator_tsv[telogator_tsv.allele_id.apply(lambda x: 'i' not in str(x))].copy()

        keepers = telogator_tsv.apply(lambda x: x['TL_p75']>p75_thresh or tvr_canon_repeats in x['tvr_consensus'],axis=1)
        telogator_tsv = telogator_tsv[keepers].copy()
        print(f'{starting_len-len(telogator_tsv)} clusters dropped for having a p75 Tel Len < 300 bp...')
        print(f"{len(telogator_tsv)} total clusters remaining!\n")
    except:
        print(f"Could not process the tsv at {path}")
    telogator_tsv.reset_index(inplace=True)
    return telogator_tsv

def get_aggregation_stats(input_df,sample_key,save_prefix,l_ratio):
 # identify proband samples that have been sequenced more than once and families
    proband_dict = {}
    family_dict = {}
    for fam_number in unique(sample_key['Patient#'].tolist()):
        tdf = sample_key[sample_key['Patient#']==fam_number]
        family_dict[fam_number] = tdf['Sample'].tolist()
        if len(tdf[tdf.Relationship.apply(lambda x: 'Proband' in x)])>1:
            proband_dict[fam_number] = tdf[tdf.Relationship.apply(lambda x: 'Proband' in x)]['Sample'].tolist()
    
    proband_df = pd.DataFrame()
    for fam in proband_dict:
        rel_cols = [x for x in input_df.columns if x.split('_')[0] in proband_dict[fam]]
        for pb_sample in proband_dict[fam]:
            comp_sample = [x for x in proband_dict[fam] if x != pb_sample][0]
            tdf = input_df[rel_cols].dropna(subset=f'{pb_sample}_tvr_consensus')
            proband_df.loc[pb_sample,'Total'] = len(tdf)
            proband_df.loc[pb_sample,'Non-Aggregated'] = sum(tdf[f'{comp_sample}_tvr_consensus'].apply(lambda x: str(x)=='nan'))
            proband_df.loc[pb_sample,'Aggregated'] = len(tdf)-sum(tdf[f'{comp_sample}_tvr_consensus'].apply(lambda x: str(x)=='nan'))
    proband_df['%Aggregated'] = proband_df['Aggregated']/proband_df['Total']*100
    proband_df['%Non-Aggregated'] = proband_df['Non-Aggregated']/proband_df['Total']*100
    proband_df.reset_index(inplace=True)
    proband_df.to_csv(save_prefix/f"3a_Proband_Aggregation_Stats_L{l_ratio}.csv",index=False)

    #check cross family alignment
    family_df = pd.DataFrame()
    for fam in family_dict:
        family_samples = family_dict[fam]
        for sample in family_samples:
            in_group = [x for x in family_samples if x!=sample]
            in_group_cols = [f'{x}_tvr_consensus' for x in in_group]
            out_group = unique([x.split('_')[0] for x in input_df.columns if 'TB' in x.split('_')[0] and x.split('_')[0] not in family_samples])
            out_group_cols = [f'{x}_tvr_consensus' for x in out_group]
            tdf = input_df.copy().dropna(subset=f'{sample}_tvr_consensus')
            family_agg = len(tdf[tdf.apply(lambda x: sum([str(x[i])!='nan' for i in in_group_cols])>0,axis=1)])
            non_family_agg = len(tdf[tdf.apply(lambda x: sum([str(x[i])!='nan' for i in out_group_cols])>0,axis=1)])
            family_df.loc[sample,'Family'] = fam
            family_df.loc[sample,'Total'] = len(tdf)
            family_df.loc[sample,'In-Family_Matches'] = family_agg
            family_df.loc[sample,'Non-Family_Matches'] = non_family_agg
            family_df.loc[sample,'In-Family_samples'] = len(in_group)
            family_df.loc[sample,'Non-Family_samples'] = len(out_group)
    family_df['%Probability_Non-Fam_Match'] = family_df['Non-Family_Matches']/family_df['Total']/family_df['Non-Family_samples']*100
    family_df.reset_index(inplace=True)
    family_df.to_csv(save_prefix/f"3b_Cross-Fam_Aggregation_Stats_L{l_ratio}.csv",index=False)
    return proband_df,family_df

###################################### Parameters to Change ##########################################################################################################
LEV_THRESHOLD = 0.85

output_dir = Path("/path/to/TelogatorOutputs")
patient_key = pd.read_excel('/path/to/PatientKey.xlsx')
analysis_dir = output_dir.parent / 'AnalysisAndFigures'
if not os.path.isdir(analysis_dir):
    os.mkdir(analysis_dir) 

rerun_status = True
################################################################################################################################################

#Start by reading in and processing all of the tsvs
all_files = [x for x in os.listdir(output_dir) if "-" in x] # change as need be to filter for Telogator2 outputs. Ensure no other outputs in output_dir that doesnt belong in PatientKey.xlsx
print(f'Using a Levenshtein ratio threshold of {LEV_THRESHOLD}...\n')
agg_order = [patient_key[patient_key['Sample']==x]['AggregationOrder'].item() for x in all_files]
sorted_pairs = sorted(zip(agg_order, all_files))
all_files = [item for _, item in sorted_pairs]

if rerun_status==True:
    print('Importing and filtering read clusters per sample...\n')
    print(f'Beginning {datetime.now()}')
    all_tsv = {}
    for file in all_files:
        all_tsv[file] = read_and_filter_tsv(output_dir/file/'tlens_by_allele.tsv',LEV_THRESHOLD,file)
    with pd.ExcelWriter(analysis_dir/f"2_Collapsed_and_Filtered_Clusters_L{int(LEV_THRESHOLD*100)}.xlsx") as writer:
        for df_name, df in all_tsv.items():
            df.to_excel(writer, sheet_name=df_name,index=False)

else:
    print('Using previous Collapsed and Filtered Clusters...\n')
    all_tsv = pd.read_excel(analysis_dir/f"2_Collapsed_and_Filtered_Clusters_L{int(LEV_THRESHOLD*100)}.xlsx",sheet_name=None)

#Start cluster aggregation!
iterator_tsvs = all_tsv.copy()
parsed = []
grouped_allele_num = 0
data_cols = ['#chr','allele_id','read_TLs','read_lengths','read_mapq','tvr_len','tvr_consensus','supporting_reads']

merged_df = pd.DataFrame()

print('Allele clusters detected per barcode...')
for x in iterator_tsvs.keys():
    print(x,len(iterator_tsvs[x]))
print('----------------------------------------')

for ref in all_files:
    print(f'Starting with sample {ref}...                   {datetime.now()}\n')
    ref_df = iterator_tsvs[ref].reset_index(drop=True)
    comparator_dfs = all_files.copy()
    comparator_dfs = [x for x in comparator_dfs if x not in parsed]
    for i in range(len(ref_df)):
        row = ref_df.loc[i]
        if str(row['tvr_consensus']) == 'nan':
            print(f'{ref}:Excluding allele with no detected tvr....')
            continue
        
        clust_list = match_clusters(row,iterator_tsvs,comparator_dfs,data_cols,LEV_THRESHOLD)
        chrom = unique([x.chrom for x in clust_list])
        if len(chrom)>1:
            chrom = "Inconsistent"
        elif len(chrom)==1:
            chrom = chrom[0]
        else:
            chrom = ''

        row_dict = {"chromosome":chrom,
                    "allele_id":grouped_allele_num,
                    "allele_ref_tvr":row['tvr_consensus'],
                    "allele_ref_dataset":ref,
                    "mean_lev_ratio":np.mean([x.lev_ratio for x in clust_list])}
        data_dict = {}
        for c in clust_list:
            data_dict = data_dict|c.data

            #remove alleles from subsequent data tables
            iterator_tsvs[c.key] = iterator_tsvs[c.key].drop(c.ind).copy()

        tdf = pd.DataFrame(row_dict|data_dict)
        merged_df = pd.concat([merged_df,tdf])

        grouped_allele_num += 1
    print(f'Done!                                                 {datetime.now()}\n')
        
    parsed.append(ref)

print('----------------------------------------')
print("Remaining alleles after merging...")
for x in iterator_tsvs.keys():
    print(x,len(iterator_tsvs[x]))

relevant_columns = [ x for x in merged_df.columns if '_tvr_consensus' in x]
merged_df['number_of_families'] = merged_df.apply(lambda x: len(unique([patient_key[patient_key.Sample==y.split('_')[0]]['Patient#'].item() for y in relevant_columns if str(x[y])!='nan'])),axis=1)

merged_df.to_csv(analysis_dir/f'3_Collapsed_Cluster_Correlation_L{int(LEV_THRESHOLD*100)}.tsv',sep='\t',index=False)
print('Calculating mapping statistics...')
proband_df,family_df = get_aggregation_stats(merged_df,patient_key,analysis_dir,int(LEV_THRESHOLD*100))
print(f'Done!  {datetime.now()}\n')

alleles_cross_aligned = len(merged_df[merged_df['number_of_families']>1])
pct_cross_aligned = round(alleles_cross_aligned/len(merged_df)*100,2)

print(f'Final number of alleles across all {len(all_files)} samples: {len(merged_df)}')
print(f'Average number of alleles detected per sample: {round(family_df.Total.mean(),2)}')
print(f'Total alleles shared by non-family members: {alleles_cross_aligned} ({pct_cross_aligned}%)')
print(f'Average proband correlation: {round(proband_df['%Aggregated'].mean(),2)}%')

# The KeyError for 'Aggregation' stats is expected as the number of families is 1. WIll not affect 3_MakePlots and 4a/b steps
