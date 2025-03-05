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

def unique(list1): #return only the unique elements of a list (no duplicates)
    x = np.array(list1)
    unique_list = np.unique(x).tolist()
    return unique_list

def str_expand(str):
    return [x for x in str]
            
def get_ordering(mut_type):
    all_types = {
    'WT':0,
    'Silent':1,
    'Alanine':2,
    'Substitution':3,
    'Deletion':4,
    'In Frame':5,
    'Frame Shift':6
    }
    if mut_type in all_types.keys():
        return all_types[mut_type]
    else:
        return max(all_types.values()) +1

def get_position(row,num_adj,wt_aa):
    if row['reclassed_mutation_type'] == 'Silent':
        return [int(re.split('(\d+)',x)[1])-num_adj-1 for x in row['mutation_name'].split(', ')]
    elif row['reclassed_mutation_type'] == 'None':
        return [-1]
    else:
        allele_aa = row['aligned_translated_adjusted_length']
        poslist =  [i for i in range(len(allele_aa)) if wt_aa[i] != allele_aa[i]]
        return poslist
    
def get_coloring(row,wt_aa_expanded):
    color_vector = [0]*len(wt_aa_expanded)
    if row['mutation_name'] == 'WT':
        return color_vector
    else:
        for i in row['positions']:
            color_vector[i] = row['ordering'] 
        for x in range(len(color_vector)):
            if row['aa_sequence_expanded'][x] == '-':
                color_vector[x] = -1
    for j in range(len(color_vector)):
        if color_vector[j]>4:
            color_vector[j]=4
    return color_vector

def make_heatmap(figure_df,save_filepath,prefix,num_adj):
    #make summary figure
    aa_numbers = [x+num_adj+1 for x in range(len(figure_df['aa_sequence_expanded'].tolist()[0]))]
    title = ' '.join(['Exon',prefix.split('_')[0],prefix.split('_')[1],'Mutations'])

    cols = [x for x in figure_df.columns.tolist() if 'Week' in x]
    colors = ['whitesmoke','grey','lightsteelblue','thistle','mistyrose','lightgrey']
    widths = [len(figure_df['color_list'].tolist()[0]),len(cols),1]
    heights = [len(figure_df)+1.5]
    gs_kw = dict(width_ratios=widths, height_ratios=heights)

    fig,ax = plt.subplots(nrows=1,ncols=3,gridspec_kw=gs_kw,figsize=((len(figure_df['color_list'].tolist()[0])+len(cols)+2)*0.3,len(figure_df)*0.3+1.5))

    cmap = mpl.colors.ListedColormap(colors)
    ax0 = sns.heatmap(figure_df['color_list'].tolist(), 
                    annot=figure_df['aa_sequence_expanded'].tolist(),
                    fmt='',cmap=cmap,
                    yticklabels=False,xticklabels=aa_numbers,
                    cbar=False,linewidths=0.5,linecolor='darkgrey',ax=ax[0],square=True)
    ax0.tick_params(axis='x', rotation=90)
    ax0.set_title(title)
    ax0.collections[0].set_clim(-1,5)   
    ax1 = sns.heatmap(figure_df[cols],
                    yticklabels=False,
                    cmap=sns.color_palette("coolwarm", as_cmap=True),
                    linewidths=0.5,linecolor='darkgrey',ax=ax[1],cbar=False,square=True)
    ax1.collections[0].set_clim(-2,2) 
    ax1.tick_params(axis='x', rotation=90)
    
    ax2 = fig.colorbar(ax[1].collections[0], cax=ax[2])
    ax2.outline.set_visible(False)

    fig.subplots_adjust(wspace=0.01)
    fig.savefig(os.path.join(save_filepath,prefix+'.png'),bbox_inches='tight')
    fig.savefig(os.path.join(save_filepath,prefix+'_Transparent.png'),bbox_inches='tight',transparent=True)
    svg_path = os.path.join(save_filepath,prefix+'.svg')
    ai_path = os.path.join(save_filepath,prefix+'.ai')
    plt.rcParams['svg.fonttype'] = 'none'
    # Save the heatmap as an SVG file
    fig.savefig(svg_path, format="svg",bbox_inches='tight')
    # Convert the SVG file to AI format
    cairosvg.svg2pdf(url=svg_path, write_to=ai_path)
    plt.close()

def establish_window(df,prot_seq):
    all_amps = unique(df.amp.tolist())
    starts = []
    ends = []
    wt_translated = ['None','Silent']
    updated_df = pd.DataFrame()
    for amp in all_amps:
        temp = df[df.amp == amp].copy()
        wt_aa = temp[temp.mutation_type.apply(lambda x: x in wt_translated)].sort_values(by='position')['translated_where_possible'].tolist()[0]
        starts = starts + [prot_seq.find(wt_aa)]
        ends = ends + [prot_seq.find(wt_aa)+len(wt_aa)]
    min_start = min(starts)
    max_end = max(ends)
    for i in range(len(all_amps)):
        front_aa = ''
        back_aa = ''
        if starts[i] > min_start:
            front_aa = prot_seq[min_start:starts[i]]
        if max_end > ends[i]:
            back_aa = prot_seq[ends[i]:max_end]
        temp = df[df.amp == all_amps[i]].copy()
        temp['translated_where_possible'] = temp.translated_where_possible.apply(lambda x: front_aa + x + back_aa)
        updated_df = pd.concat([updated_df,temp])
    return updated_df

def align_allele(wt_aa,row):
    allele_aa = row['translated_where_possible']
    if len(allele_aa) == len(wt_aa):
        return allele_aa
    elif len(allele_aa) < len(wt_aa):
        mut_pos = [i for i in range(len(allele_aa)) if allele_aa[i]!=wt_aa[i]]
        if len(mut_pos) == 0:
            aligned_allele = allele_aa + ''.join(['-']*(len(wt_aa)-len(allele_aa)))
            return aligned_allele
        elif row['reclassed_mutation_type'] == 'Deletion':
            aligned_allele = allele_aa[:mut_pos[0]] + '-' + allele_aa[mut_pos[0]:]
            return aligned_allele
        elif row['reclassed_mutation_type'] == 'In Frame':
            dashes_to_add = int(int(row['mutation_name'].split('-')[-1][:-2])/3)
            aligned_allele = allele_aa[:mut_pos[0]] + ''.join(['-']*dashes_to_add) + allele_aa[mut_pos[0]:]
            return aligned_allele
        else:
            aligned_allele = allele_aa + ''.join(['-']*(len(wt_aa)-len(allele_aa)))
            return aligned_allele
    elif len(allele_aa) > len(wt_aa):
        return allele_aa[0:len(wt_aa)]

combined_df = pd.read_csv('/data/annika/POT1_Screen_VUS_Combined/FigureGeneration/PersistenceDepletion/1_1e-06_Combined_3Bioreps_scored.csv')
save_base_filepath = '/data/annika/POT1_Screen_VUS_Combined/FigureGeneration/Heatmaps'
if os.path.exists(save_base_filepath) == False:
        os.mkdir(save_base_filepath)

with open(r"/home/annika/Documents/SequencingAnalysis/LoHapScreening/Re_Running_POT1_Screen/HDR_Screen/Input/POT1_Prot.txt", "r") as file:
    prot_seq = file.read().upper()

for exon in unique(combined_df.exon.tolist()):
    try:
        save_filepath = os.path.join(save_base_filepath,str(exon))
        if os.path.exists(save_filepath) == False:
            os.mkdir(save_filepath)

        tdf = combined_df[combined_df.exon==exon].copy()
        if len(unique(tdf.amp.tolist()))>1:
            tdf = establish_window(tdf,prot_seq)

        wt_aa = unique(tdf[tdf['mutation_name'] == 'WT']['translated_where_possible'].tolist())
        if len(wt_aa)==1:
            wt_aa = wt_aa[0]
            wt_len = len(wt_aa)

            figure_df = pd.DataFrame()
            cols = [x for x in tdf.columns.tolist() if 'Log2FC' in x and 'mean' not in x]
            figure_df = tdf[['translated_where_possible','mutation_name','position','reclassed_mutation_type','exp']+cols].copy()
            figure_df['aligned_translated_adjusted_length'] = figure_df.apply(lambda x: align_allele(wt_aa,x), axis = 1)

            if len(unique(figure_df.aligned_translated_adjusted_length.apply(lambda x: len(x)).tolist()))>1:
                print('Issue with Allele Alignment on Exon %d' % exon)
                break

            figure_df = figure_df.dropna(subset=['mutation_name','reclassed_mutation_type'])
            figure_df['aa_sequence_expanded'] = figure_df.aligned_translated_adjusted_length.apply(lambda x: str_expand(x))
            figure_df['first_mut'] = figure_df['position']
            figure_df = figure_df.drop(['translated_where_possible','position'],axis=1)
            num_adj = prot_seq.find(wt_aa)
            figure_df['positions'] = figure_df.apply(lambda x: get_position(x,num_adj,wt_aa),axis=1)

            figure_df['ordering'] = figure_df['reclassed_mutation_type'].apply(get_ordering)
            figure_df = figure_df.sort_values(by=['ordering','first_mut','mutation_name'])
            wt_aa_expanded = str_expand(wt_aa)
            
            colors = []
            for index, row in figure_df.iterrows():
                colors = colors + [get_coloring(row,wt_aa_expanded)]
            figure_df['color_list'] = colors

            #Make Summary Figure
            make_heatmap(figure_df,save_filepath,str(exon)+'_Summary',num_adj)
            figure_df.to_excel(os.path.join(save_filepath,str(exon)+'_Figure.xlsx'))

            #Look only at targeted mutations
            mutation_types = ['Alanine','Silent','Substitution','Deletion']
            tdf = pd.DataFrame()
            tdf = pd.concat([tdf,figure_df[figure_df['mutation_name']=='WT'].copy()])
            tdf = pd.concat([tdf,figure_df[[x in mutation_types for x in figure_df['reclassed_mutation_type'].tolist()]].copy()])
            make_heatmap(tdf,save_filepath,str(exon)+'_Targeted',num_adj)

            #Just Alanines and Substitutions
            missense_muts = ['Alanine','Substitution']
            tdf = pd.DataFrame()
            tdf = pd.concat([tdf,figure_df[figure_df['mutation_name']=='WT'].copy()])
            tdf = pd.concat([tdf,figure_df[[x in missense_muts for x in figure_df['reclassed_mutation_type'].tolist()]].copy()])
            tdf = tdf.sort_values(by=['first_mut','reclassed_mutation_type'])
            make_heatmap(tdf,save_filepath,str(exon)+'_Missense',num_adj)

            #Collapse Bioreps 
            cols = [x for x in figure_df.keys().tolist() if 'Pool' not in x] 
            collapsed_df = figure_df[cols].copy()
            
            pools = [x for x in figure_df.keys().tolist() if 'Pool' in x] 
            new_pools = ['_'.join([x.split('_')[0],'Mean',x.split('_')[2]]) for x in pools]
            tdf = figure_df[pools].copy()
            tdf = tdf.rename(columns=dict(map(lambda i,j : (i,j) , pools,new_pools)))
            tdf_mean = tdf.groupby(by= tdf.columns, axis=1).mean()
            
            tdf_std = tdf.groupby(by= tdf.columns, axis=1).mean()
            pools = [x for x in tdf.keys().tolist() if 'Week' in x] 
            new_pools = [x.replace('Mean','Std') for x in pools]
            tdf_std = tdf_std.rename(columns=dict(map(lambda i,j: (i,j),pools,new_pools)))
            collapsed_df_summary = pd.concat([collapsed_df,tdf_mean,tdf_std],axis=1)
            collapsed_df_summary.to_excel(os.path.join(save_filepath,str(exon)+'_Wk3Mean.xlsx'))

            collapsed_df_figure = pd.concat([collapsed_df,tdf_mean],axis=1)
            make_heatmap(collapsed_df_figure,save_filepath,str(exon)+'_Summary_Collapsed',num_adj)

            #Look only at targeted mutations
            mutation_types = ['Alanine','Silent','Substitution','Deletion']
            tdf = pd.DataFrame()
            tdf = pd.concat([tdf,collapsed_df_figure[figure_df['mutation_name']=='WT'].copy()])
            tdf = pd.concat([tdf,collapsed_df_figure[[x in mutation_types for x in collapsed_df_figure['reclassed_mutation_type'].tolist()]].copy()])
            make_heatmap(tdf,save_filepath,str(exon)+'_Targeted_Collapsed',num_adj)

            #Just Alanines and Substitutions
            missense_muts = ['Alanine','Substitution']
            tdf = pd.DataFrame()
            tdf = pd.concat([tdf,collapsed_df_figure[figure_df['mutation_name']=='WT'].copy()])
            tdf = pd.concat([tdf,collapsed_df_figure[[x in missense_muts for x in figure_df['reclassed_mutation_type'].tolist()]].copy()])
            tdf = tdf.sort_values(by=['first_mut','reclassed_mutation_type'])
            make_heatmap(tdf,save_filepath,str(exon)+'_Missense_Collapsed',num_adj)

        else:
            print('Error with Exon %d: WT alleles do not agree!' % exon)
    except:
        print('Failed to make heatmap for exon %d' % exon)

    