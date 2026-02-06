import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import cairosvg
import os
from itertools import groupby,combinations
import random
from datetime import datetime
import statsmodels.api as sm


def unique(list1):
    '''return only the unique elements of a list (no duplicates)'''
    x = np.array(list1)
    unique_list = np.unique(x).tolist()
    return unique_list

def find_average(values_list,operator):
    if operator == 'mean':
        return np.mean(values_list)
    if operator == '75pct':
        return np.percentile(values_list,75)
    if operator == 'median':
        return np.median(values_list)
    else:
        print('Operator not recognized!')
        return
    
def get_tel_len_list(read_string,adjust=False,tvr_len=0):
    if str(read_string)=='nan':
        return np.nan
    read_list = [int(x) for x in read_string.split(',')]
    if adjust:
        read_list = [x+tvr_len for x in read_list]
    return read_list
    
def rename_allele_ids_by_length(dataframe,adjust,sampleID,average_string):
    dataframe['allele_order'] = dataframe.apply(lambda x: find_average(
                                           values_list=get_tel_len_list(read_string=x[f'{sampleID}_read_TLs'],
                                           adjust=adjust,
                                           tvr_len=x[f'{sampleID}_tvr_len']),
                                           operator=average_string),
                                           axis=1).tolist()
    dataframe = dataframe.sort_values(by='allele_order').reset_index(drop=True)
    dataframe['plot_allele_id']= [x for x in range(len(dataframe))]
    return dataframe

def get_carrier_classification(sample_df,family_df,carrier,noncarrier):
    carrier_dict = {}
    noncarrier_dict = {}
    if str(carrier)==str(noncarrier)=='nan':
        sample_df['allele_class'] = 'Uncertain'
        return sample_df
    if str(carrier)!='nan':
        carrier_alleles = family_df[family_df[f'{carrier}_read_TLs'].apply(lambda x: len(str(x).split(','))>1)].plot_allele_id.tolist()
        carrier_dict = dict(zip(carrier_alleles,['Carrier']*len(carrier_alleles)))
    if str(noncarrier)!='nan':
        noncarrier_alleles = family_df[family_df[f'{noncarrier}_read_TLs'].apply(lambda x: len(str(x).split(','))>1)].plot_allele_id.tolist()
        noncarrier_dict = dict(zip(noncarrier_alleles,['NonCarrier']*len(noncarrier_alleles)))
    combined_dict = carrier_dict|noncarrier_dict
    cross_aligned_keys = [x for x in carrier_dict.keys() if x in noncarrier_dict.keys()]
    if len(cross_aligned_keys)>0:
        for x in cross_aligned_keys:
            combined_dict[x] = 'Uncertain'
    sample_df['allele_class'] = sample_df['plot_allele_id'].apply(lambda x: combined_dict[x] if x in combined_dict.keys() else 'Uncertain')
    return sample_df

def get_parental_tl(row,reference_df,carrier,noncarrier,average_string,adjust):
    allele = row['plot_allele_id']
    if row['allele_class']=='Uncertain':
        return None
    elif row['allele_class']=='Carrier':
        allele_str = reference_df[reference_df.plot_allele_id==allele][f'{carrier}_read_TLs'].item()
        tvr_len = reference_df[reference_df.plot_allele_id==allele][f'{carrier}_tvr_len'].item()
    elif row['allele_class']=='NonCarrier':
        allele_str = reference_df[reference_df.plot_allele_id==allele][f'{noncarrier}_read_TLs'].item()
        tvr_len = reference_df[reference_df.plot_allele_id==allele][f'{noncarrier}_tvr_len'].item()
    return find_average(get_tel_len_list(allele_str,adjust,tvr_len),average_string)

def get_stripplot_data(row):
    df = pd.DataFrame(columns=["plot_allele_id",'order','total_len','allele_class'])

    total_len = row['total_len']
    order = [row['order']]*len(total_len)
    allele_id = [row['plot_allele_id']]*len(total_len)
    allele_class = [row['allele_class']]*len(total_len)

    df['plot_allele_id'] = allele_id
    df['order'] = order
    df['total_len'] = total_len
    df['allele_class'] = allele_class

    return df

def compare_two_seq(dataframe,condition1,condition2,save_path,file_prefix,average_method,
                    r2_box=True,xlabel='',ylabel='',title='',xmin=None,xmax=None,ymin=None,ymax=None):
    
    dataframe = dataframe.dropna(subset=[condition1,condition2])

    if xlabel=='':
        xlabel = condition1.split("_")[0]
    if ylabel=='':
        ylabel = condition2.split("_")[0]
    if title=='':
        title = f'{average_method.capitalize()} Telomere Length {xlabel} vs {ylabel}'
    
    X = dataframe[condition1].tolist()
    Y = dataframe[condition2].tolist()
    const_x = sm.add_constant(X, prepend=False)
    model = sm.OLS(Y, const_x)
    results = model.fit()
    slope,intercept = results.params
    r_squared = results.rsquared_adj

    line_x = [x for x in range(round(min(X)),round(max(X)),round((max(X)-min(X))/10))] + [max(X)]
    line_y = [slope*x+intercept for x in line_x]

    confidence_intervals = results.conf_int(alpha=0.05)
    lower_bounds = [x * confidence_intervals[0,0] + confidence_intervals[1,0] for x in line_x]
    upper_bounds = [x * confidence_intervals[0,1] + confidence_intervals[1,1] for x in line_x]

    fig,ax = plt.subplots(figsize = (5,5))
    i = sns.scatterplot(data=dataframe,x=condition1,y=condition2)
    j = plt.plot(line_x,line_y)
    k = plt.fill_between(line_x,lower_bounds,upper_bounds,color='black',alpha=0.1)

    if r2_box:
        text_content = f"Slope: {round(slope,4)}\nR2: {round(r_squared,4)}"
        ax.text(0.95, 0.05, text_content, transform=ax.transAxes, ha='right', va='bottom',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin,xmax)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin,ymax)
        
    fig.savefig(save_path/f'{file_prefix}_scatter.png',bbox_inches='tight')
    fig.savefig(save_path/f'{file_prefix}_scatter_transparent.png',bbox_inches='tight',transparent=True)
    svg_path = save_path/f'{file_prefix}_scatter.svg'
    ai_path = save_path/f'{file_prefix}_scatter.ai'
    plt.rcParams['svg.fonttype'] = 'none'
    # Save the figure as an SVG file
    fig.savefig(svg_path, format="svg",bbox_inches='tight')
    # Convert the SVG file to AI format
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(ai_path))
    plt.show()
    plt.close()

    with open(save_path/f'{file_prefix}_regression_stats.txt', 'w') as f:
        f.write(results.summary().as_text())

    return

def make_rank_ordered_barplot(dataframe,name,average_string,color_order,plot_palette,save_filepath,
                              overlay_parental=False):
    stripplot_df = pd.concat(dataframe.apply(lambda x: get_stripplot_data(x),axis=1).tolist())
    fig,ax = plt.subplots(figsize = (21,7))
    sample_id = unique([x.split('_')[0] for x in dataframe.columns if 'TB-' in x])[0]
    
    if overlay_parental:
        save_prefix = f'{sample_id}_Rank_Ordered_{average_string}_TL_Parental'
        i = sns.barplot(dataframe,x='order',y=f'{average_string}_len',color=plot_palette[2],ax=ax)
        k = sns.barplot(dataframe,x='order',y=f'parental_{average_string}_len',hue='allele_class',
                        hue_order=color_order,palette=plot_palette,ax=ax)
    else:
        save_prefix = f'{sample_id}_Rank_Ordered_{average_string}_TL'
        i = sns.barplot(dataframe,x='order',y=f'{average_string}_len',hue='allele_class',
                        hue_order=color_order,palette=plot_palette,ax=ax)
    j = sns.stripplot(stripplot_df,x='order',y='total_len',ax=ax,color='black',size=2,jitter=True)
    plt.legend(loc='upper left') 
    ax.set_ylim(0,20000)
    ax.set_xticks(range(len(dataframe)))
    ax.set_xticklabels(dataframe.plot_allele_id.tolist())
    ax.set_xlabel('Allele ID')
    plt.xticks(rotation=45)
    ax.set_ylabel('Telomere Length')
    ax.set_title(f"Rank Ordered {average_string.capitalize()} Telomere Lengths: {name}")
    fig.savefig(save_filepath/f'{save_prefix}.png',bbox_inches='tight')
    fig.savefig(save_filepath/f'{save_prefix}_transparent.png',bbox_inches='tight',transparent=True)
    svg_path = save_filepath/f'{save_prefix}.svg'
    ai_path = save_filepath/f'{save_prefix}.ai'
    plt.rcParams['svg.fonttype'] = 'none'
    # Save the figure as an SVG file
    fig.savefig(svg_path, format="svg",bbox_inches='tight')
    # Convert the SVG file to AI format
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(ai_path))
    plt.close()

def split_repeated_characters(s):
    """
    Splits a string into a list of substrings, where each substring
    consists of consecutive, identical characters.

    Args:
        s (str): The input string.

    Returns:
        list: A list of substrings.
    """
    result = []
    for key, group in groupby(s):
        result.append("".join(group))
    return result

def get_parental_color(x,colors_list):
    if x=='Carrier':
        return colors_list[0]
    elif x=='NonCarrier':
        return colors_list[1]
    else:
        return colors_list[2]

def make_patient_specific_stacked_bar_chart(dataframe,colors_dict,name,parental_colors,sample,save_filepath,avg_operator):
    save_prefix = f'{sample}_{avg_operator}_stacked_bar_all_alleles'
    
    fig,ax = plt.subplots(figsize = (15,0.5*len(dataframe)))
    stripplot_df = pd.concat(dataframe.apply(lambda x: get_stripplot_data(x),axis=1).tolist())
    stripplot_df['order_name'] = stripplot_df['order'].apply(lambda x: str(x))
    dataframe['order_name'] = dataframe.order.apply(lambda x: str(x))
    i=sns.stripplot(data=stripplot_df,y='order_name',x='total_len',color='black',size=2,jitter=True)
    j=sns.barplot(data=dataframe,y='order_name',x=f'{avg_operator}_len',color='lightgrey')
    for allele in dataframe['order'].tolist():
        tvr_sequence = dataframe[dataframe['order']==allele][f'{sample}_tvr_consensus'].item()
        if str(tvr_sequence)=='nan':
                continue
        tvr_sequence = split_repeated_characters(tvr_sequence)
        repeats = [x[0] for x in tvr_sequence]
        lengths = [len(x) for x in tvr_sequence]
        zero = 0
        for i in range(len(repeats)):
            plt.barh(y=[allele],width=lengths[i],color=colors_dict[repeats[i]],left=zero)
            zero+=lengths[i]
        
        plt.barh(y=[allele],width=600,left=-600,
                color=get_parental_color(dataframe[dataframe['order']==allele][f'allele_class'].item(),parental_colors))
        
    plt.ylim([-1,len(dataframe)])
    plt.xlim([-600,20000])
    plt.axvline(x=0, color='black',linewidth=0.75)
    ax.set_xlabel('Telomere Length')
    ax.set_ylabel('Allele ID')
    ax.set_yticks(range(len(dataframe)))
    ax.set_yticklabels(dataframe.plot_allele_id.tolist())
    ax.set_title(f'{name}: All Alleles TVR Visualization')
    fig.savefig(save_filepath/f'{save_prefix}.png',bbox_inches='tight')
    fig.savefig(save_filepath/f'{save_prefix}_transparent.png',bbox_inches='tight',transparent=True)
    svg_path = save_filepath/f'{save_prefix}.svg'
    ai_path = save_filepath/f'{save_prefix}.ai'
    plt.rcParams['svg.fonttype'] = 'none'
    # Save the figure as an SVG file
    fig.savefig(svg_path, format="svg",bbox_inches='tight')
    # Convert the SVG file to AI format
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(ai_path))
    plt.close()

def stacked_bar_plotter(bar_df,strip_df,colors_dict,patient_key,familiy_num,allele_num,avg_operator,save_filepath):
    fig,ax = plt.subplots(figsize = (15,0.5*len(bar_df)))
    i=sns.stripplot(data=strip_df,y='order_name',x='total_len',color='black',size=2,jitter=True)
    j=sns.barplot(data=bar_df,y='order_name',x='tel_len',color='lightgrey')
    for name in bar_df.order_name.tolist():
        tvr_sequence = bar_df[bar_df.order_name==name]['tvr_seq'].item()
        if str(tvr_sequence)=='nan':
            continue
        repeats = [x[0] for x in tvr_sequence]
        lengths = [len(x) for x in tvr_sequence]
        zero = 0
        for i in range(len(repeats)):
            plt.barh(y=[name],width=lengths[i],color=colors_dict[repeats[i]],left=zero)
            zero+=lengths[i]
    ax.set_title(f'Patient Family {familiy_num}: Allele {allele_num}')
    ax.set_yticks(range(len(bar_df)))
    labels = [patient_key[patient_key['Sample']==x]['FigureName'].item() for x in bar_df['sample'].tolist()]
    ax.set_yticklabels(labels)
    ax.set_xlabel(f'{avg_operator}Telomere Length')
    ax.set_ylabel('Sample Name')

    save_prefix = f'Allele{allele_num}_{avg_operator}_stacked_bar'
    fig.savefig(save_filepath/f'{save_prefix}.png',bbox_inches='tight')
    fig.savefig(save_filepath/f'{save_prefix}_transparent.png',bbox_inches='tight',transparent=True)
    svg_path = save_filepath/f'{save_prefix}.svg'
    ai_path = save_filepath/f'{save_prefix}.ai'
    plt.rcParams['svg.fonttype'] = 'none'
    # Save the figure as an SVG file
    fig.savefig(svg_path, format="svg",bbox_inches='tight')
    # Convert the SVG file to AI format
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(ai_path))
    plt.close()
    return

def make_allele_stacked_bar_charts(majority_alleles,family_df,fam_samples,fam_cols,avg_operator,patient_key,colors_dict,family_num,save_filepath,add_tvr_len_to_TL):
    for allele_id in majority_alleles:
        allele_strip_df = []
        barplot_data = [] 
        ind = family_df[family_df.plot_allele_id==allele_id].index
        row = family_df.loc[ind]
        for patient_sample in fam_samples:
            patient_cols = [x for x in fam_cols if patient_sample in x]+['plot_allele_id']
            patient_df = row[patient_cols].copy()
            patient_df = patient_df.dropna(subset=f'{patient_sample}_tvr_len')
            if len(patient_df)==0:
                continue
            patient_df['total_len'] = patient_df.apply(lambda x: 
                                                    get_tel_len_list(read_string=x[f'{patient_sample}_read_TLs'],
                                                                        adjust=add_tvr_len_to_TL,
                                                                        tvr_len=x[f'{patient_sample}_tvr_len']),axis=1)
            patient_df[f'{avg_operator}_len'] = patient_df['total_len'].apply(lambda x: find_average(x,avg_operator))
            patient_df['order'] = patient_key[patient_key['Sample']==patient_sample]['FigureOrder'].item()
            patient_df['allele_class'] = 'Shared'
            temp_strip_df = get_stripplot_data(patient_df.loc[ind].to_dict(orient='records')[0])
            temp_strip_df['sample'] = patient_sample
            allele_strip_df.append(temp_strip_df)
            barplot_data.append([patient_sample,patient_df[f'{patient_sample}_tvr_consensus'].item(),patient_df[f'{avg_operator}_len'].item(),
                                patient_key[patient_key['Sample']==patient_sample]['FigureOrder'].item()])
        barplot_data = pd.DataFrame(data=barplot_data,columns=['sample','tvr_seq','tel_len','order'])
        barplot_data = barplot_data.sort_values(by='order').reset_index(drop=True)
        allele_strip_df = pd.concat(allele_strip_df)
        allele_strip_df = allele_strip_df.sort_values(by='order').reset_index(drop=True)
        allele_strip_df['order_name'] = allele_strip_df.order.apply(lambda x: str(x))
        barplot_data['order_name'] = barplot_data.order.apply(lambda x: str(x))
        stacked_bar_plotter(barplot_data,allele_strip_df,colors_dict,patient_key,family_num,allele_id,avg_operator,save_filepath)
    return

def get_proband_sample_info_for_comp(proband_data,ave_operator,save_path,fam_num):
    proband_dataframe = []
    for pb_sample in proband_data.keys():
        tdf = proband_data[pb_sample]
        tdf[f'{pb_sample}_{ave_operator}_len'] = tdf[f'{ave_operator}_len']
        tdf[f'{pb_sample}_total_len'] = tdf[f'total_len']
        tdf[f'{pb_sample}_allele_class'] = tdf[f'allele_class']
        proband_dataframe.append(tdf[['plot_allele_id',f'{pb_sample}_{ave_operator}_len',
                                        f'{pb_sample}_supporting_reads',
                                        f'{pb_sample}_total_len']].set_index('plot_allele_id'))
    proband_dataframe = pd.concat(proband_dataframe,axis=1)
    proband_dataframe.to_csv(save_path/f'Probands_Family{fam_num}_{ave_operator}.tsv',sep='\t',index=False)
    rel_cols = [x for x in proband_dataframe.columns if f'{ave_operator}_len' in x]
    save_prefix = f"Family{fam_num}_Proband_{ave_operator}_Cluster_Comp"
    compare_two_seq(proband_dataframe,rel_cols[0],rel_cols[1],save_path,save_prefix,ave_operator)
    
    save_prefix = f"Family{fam_num}_Proband_{ave_operator}_Cluster_Comp_Zeros"
    proband_dataframe = proband_dataframe.fillna(0)
    compare_two_seq(proband_dataframe,rel_cols[0],rel_cols[1],save_path,save_prefix,ave_operator)
    return

def make_scatter_plots(xval,yval,dataframe,colors,order,save_filepath,xlabel,ylabel,title,save_prefix,r2_box=False,
                       xmin=None,xmax=None,ymin=None,ymax=None):
    
    with open(save_filepath/f'{save_prefix}_regression_stats.txt', 'w') as f:
        pass

    fig,ax = plt.subplots(figsize = (5,5))
    rval_dict = {}
    line_x = [x for x in range(round(dataframe[xval].min()),round(dataframe[xval].max()),round((dataframe[xval].max() - dataframe[xval].min())/10))] + [round(dataframe[xval].max())]
    for allele_class in unique(dataframe.allele_class.tolist()):
        temp_df = dataframe[dataframe.allele_class == allele_class][[xval,yval]]
        temp_df = temp_df.dropna()
        if len(temp_df)<=0:
            continue

        X = temp_df[xval].tolist()
        Y = temp_df[yval].tolist()

        const_x = sm.add_constant(X, prepend=False)
        model = sm.OLS(Y, const_x)
        results = model.fit()
        slope,intercept = results.params
        r_squared = results.rsquared_adj

        if xmin is not None and xmax is not None:
            line_x = [x for x in range(round(xmin),round(xmax),round((xmax-xmin)/10))]+[xmax]

        line_y = [slope*x+intercept for x in line_x]

        confidence_intervals = results.conf_int(alpha=0.05)
        lower_bounds = [x * confidence_intervals[0,0] + confidence_intervals[1,0] for x in line_x]
        upper_bounds = [x * confidence_intervals[0,1] + confidence_intervals[1,1] for x in line_x]


        i = sns.scatterplot(data=temp_df,x=xval,y=yval,
                            color=colors[order.index(allele_class)],ax=ax)
        j = plt.plot(line_x,line_y,color=colors[order.index(allele_class)])
        k = plt.fill_between(line_x,lower_bounds,upper_bounds,color=colors[order.index(allele_class)],
                             alpha=0.1)
        with open(save_filepath/f'{save_prefix}_regression_stats.txt', 'a') as f:
            f.write(f'{allele_class} Allele Regression:\n')
            f.write(results.summary().as_text())
            f.write('\n\n'+'='*78+'\n'+'='*78+'\n\n')

        rval_dict[allele_class] = r_squared
        
    if r2_box:
        text_content = '\n'.join([f'{x} R2: {round(rval_dict[x],3)}' for x in rval_dict.keys()])
        ax.text(0.95, 0.05, text_content, transform=ax.transAxes, ha='right', va='bottom',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin,xmax)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin,ymax)
    fig.savefig(save_filepath/f'{save_prefix}.png',bbox_inches='tight')
    fig.savefig(save_filepath/f'{save_prefix}_transparent.png',bbox_inches='tight',transparent=True)
    svg_path = save_filepath/f'{save_prefix}.svg'
    ai_path = save_filepath/f'{save_prefix}.ai'
    plt.rcParams['svg.fonttype'] = 'none'
    # Save the figure as an SVG file
    fig.savefig(svg_path, format="svg",bbox_inches='tight')
    # Convert the SVG file to AI format
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(ai_path))
    plt.close()
    return

def parental_comp_scatter_plot(dataframe,operator,colors,order,save_filepath,sample,patient_name,r2_box=True):
    
    xval = f'parental_{operator}_len'
    yval = f'{operator}_len'
    xlabel = 'Parental Telomere Length'
    ylabel = 'Proband Telomere Length'
    title = f'{patient_name} Parental Telomere Length Correlation'
    save_prefix = f'{sample}_parental_{operator}_tel_len_scatter'
    make_scatter_plots(xval,yval,dataframe,colors,order,save_filepath,xlabel,ylabel,title,save_prefix,r2_box)
    
    dataframe['tel_len_change'] = dataframe[yval]-dataframe[xval]
    yval = 'tel_len_change'
    ylabel = 'Change Relative to Proband'
    title = f'{patient_name} Delta Telomere Length Correlation'
    save_prefix = f'{sample}_{operator}_change_tel_len_scatter'
    make_scatter_plots(xval,yval,dataframe,colors,order,save_filepath,xlabel,ylabel,title,save_prefix,r2_box)

    xval = f'parental_rank_order'
    yval = 'inherited_rank_order'
    xlabel = f'Rank Order of {operator.capitalize()} Parental Telomere Length'
    ylabel = f'Rank Order of {operator.capitalize()} Patient Telomere Length'
    title = f'{patient_name} Rank Order Comparison'
    save_prefix = f'{sample}_{operator}_parental_rank_scatter'
    make_scatter_plots(xval,yval,dataframe,colors,order,save_filepath,xlabel,ylabel,title,save_prefix,r2_box)

    return

def make_skew_barplots(skew_df,figname,id,base_filepath,colors,order,method):
    skew_df.loc[0,'grouping_method']='All'
    grouping_dict = {'halves':2,'thirds':3,'quartiles':4}
    skew_df['%Total'] = skew_df['%Carrier']+skew_df['%Other']
    for group_metric in grouping_dict.keys():
        save_filepath = base_filepath/group_metric.capitalize()
        if not os.path.isdir(save_filepath):
            os.mkdir(save_filepath)
        all_groups = ['All',group_metric]
        temp_df = skew_df[skew_df.grouping_method.apply(lambda x: x in all_groups)]
        fig,ax = plt.subplots(figsize = (1*len(temp_df),5))
        sns.barplot(data=temp_df,x='group',y='%Total',color=colors[order.index('Uncertain')])
        sns.barplot(data=temp_df,x='group',y='%Carrier',color=colors[order.index('Carrier')])
        plt.axvline(x=1.5, color='black', linestyle='--')
        ax.set_xlabel('Group Subsampled')
        ax.set_ylabel('% Carrier Inherited Alleles')
        ax.set_title(f'{figname} {group_metric.capitalize()} Carrier Allele Enrichment')

        save_prefix = f'{id}_{method}_{group_metric}_allele_prop_bar'

        fig.savefig(save_filepath/f'{save_prefix}.png',bbox_inches='tight')
        fig.savefig(save_filepath/f'{save_prefix}_transparent.png',bbox_inches='tight',transparent=True)
        svg_path = save_filepath/f'{save_prefix}.svg'
        ai_path = save_filepath/f'{save_prefix}.ai'
        plt.rcParams['svg.fonttype'] = 'none'
        # Save the figure as an SVG file
        fig.savefig(svg_path, format="svg",bbox_inches='tight')
        # Convert the SVG file to AI format
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(ai_path))
        plt.close()
    return

def get_patient_inheritence_skew(sample_df,patient_ID,patient_figname,save_filepath,colors,order,method,patient_key):
    patient_skew = []
    num_carrier = sum(sample_df.allele_class=='Carrier')
    num_other = len(sample_df)-num_carrier
    total_alleles = len(sample_df)
    patient_skew.append([patient_ID,num_carrier/total_alleles*100,num_other/total_alleles*100,
                        num_carrier,num_other,'All','A',
                        patient_key.set_index('Sample').loc[patient_ID,'InheritancePattern'],
                        patient_key.set_index('Sample').loc[patient_ID,'caPOT1Status']])
    group_dict = {2:'halves',3:'thirds',4:'quartiles'}
    for num_groups in group_dict.keys():
        #random sample three times
        group_num_alleles = int(np.ceil(total_alleles/num_groups))
        for i in range(3):
            random_inds = random.sample(sample_df['order'].tolist(), group_num_alleles)
            num_carrier = sample_df.loc[random_inds]['allele_class'].tolist().count('Carrier')
            num_other = len(sample_df.loc[random_inds])-num_carrier
            group_total_alleles = len(sample_df.loc[random_inds])
            patient_skew.append([patient_ID,num_carrier/group_total_alleles*100,
                                    num_other/group_total_alleles*100,
                                    num_carrier,num_other,group_dict[num_groups],'R',
                                    patient_key.set_index('Sample').loc[patient_ID,'InheritancePattern'],
                                    patient_key.set_index('Sample').loc[patient_ID,'caPOT1Status']])
        
        counter = 1
        for x in range(0,total_alleles,group_num_alleles):
            if counter<num_groups:
                inds = [x for x in range(x,x+group_num_alleles)]
            else:
                inds = [x for x in range(x,len(sample_df))]
            num_carrier = sample_df.loc[inds]['allele_class'].tolist().count('Carrier')
            num_other = len(sample_df.loc[inds])-num_carrier
            group_total_alleles = len(sample_df.loc[inds])
            patient_skew.append([patient_ID,num_carrier/group_total_alleles*100,
                                num_other/group_total_alleles*100,
                                num_carrier,num_other,group_dict[num_groups],str(counter),
                                patient_key.set_index('Sample').loc[patient_ID,'InheritancePattern'],
                                patient_key.set_index('Sample').loc[patient_ID,'caPOT1Status']])
            counter+=1

    patient_skew = pd.DataFrame(data=patient_skew,columns=['sample','%Carrier','%Other','#Carrier',
                                                            '#Other','grouping_method','group','inheritance','mutation_status'])
    
    make_skew_barplots(patient_skew,patient_figname,patient_ID,save_filepath,colors,order,method)
    return patient_skew

def normalize_to_non_subsetted(row,input_df):
    tdf = input_df[input_df['method']==row['method']]
    tdf = tdf[tdf['sample']==row['sample']]
    tdf = tdf[tdf['grouping_method']=='All']
    normed_carrier = row['%Carrier']/tdf['%Carrier'].item()
    normed_non_carrier = row['%Other']/tdf['%Other'].item()
    return normed_carrier,normed_non_carrier

def make_skew_summary_plots(base_skew_df,save_filepath,colors,hue_order):
    for mean_metric in unique(base_skew_df.method.tolist()):
        skew_df = base_skew_df[base_skew_df.method==mean_metric]
        grouping_dict = {'halves':2,'thirds':3,'quartiles':4}

        for group_metric in grouping_dict.keys():
            all_groups = ['All',group_metric]
            temp_df = skew_df[skew_df.grouping_method.apply(lambda x: x in all_groups)]

            ####################################################################################################################
            fig,ax = plt.subplots(figsize = (0.75*(grouping_dict[group_metric]+2),4))
            sns.stripplot(data=temp_df,x='group',y='%Carrier',color='black',size=4.5,zorder=0)
            sns.pointplot(data=temp_df,x='group',y='%Carrier',errorbar='sd',capsize=.2,
                        linestyle="none",color='firebrick',linewidth=2,estimator=np.median,
                        marker="_", markersize=20, markeredgewidth=3,)
            plt.axvline(x=1.5, color='black', linestyle='--')
            ax.set_xlabel('Group')
            ax.set_ylabel('Percent Carrier Inherited Telomeres')
            ax.set_title(f'Allele Enrichment ({mean_metric.capitalize()} Rank Ordered)')
            file_prefix = f'{mean_metric}_allele_enrichment_{group_metric}'
            fig.savefig(save_filepath/f'{file_prefix}.png',bbox_inches='tight')
            fig.savefig(save_filepath/f'{file_prefix}_transparent.png',bbox_inches='tight',transparent=True)
            svg_path = save_filepath/f'{file_prefix}.svg'
            ai_path = save_filepath/f'{file_prefix}.ai'
            plt.rcParams['svg.fonttype'] = 'none'
            # Save the figure as an SVG file
            fig.savefig(svg_path, format="svg",bbox_inches='tight')
            # Convert the SVG file to AI format
            cairosvg.svg2pdf(url=str(svg_path), write_to=str(ai_path))
            plt.close()

            fig,ax = plt.subplots(figsize = (0.75*(grouping_dict[group_metric]+2),4))
            sns.stripplot(data=temp_df,x='group',y='Normalized%Carrier',color='black',size=4.5,zorder=0)
            sns.pointplot(data=temp_df,x='group',y='Normalized%Carrier',errorbar='sd',capsize=.2,
                        linestyle="none",color='firebrick',linewidth=2,estimator=np.median,
                        marker="_", markersize=20, markeredgewidth=3,)
            plt.axvline(x=1.5, color='black', linestyle='--')
            ax.set_xlabel('Group')
            ax.set_ylabel('Normalized %Carrier Inherited Telomeres')
            ax.set_title(f'Allele Enrichment ({mean_metric.capitalize()} Rank Ordered)')
            file_prefix = f'{mean_metric}_allele_enrichment_{group_metric}_normalized'
            fig.savefig(save_filepath/f'{file_prefix}.png',bbox_inches='tight')
            fig.savefig(save_filepath/f'{file_prefix}_transparent.png',bbox_inches='tight',transparent=True)
            svg_path = save_filepath/f'{file_prefix}.svg'
            ai_path = save_filepath/f'{file_prefix}.ai'
            plt.rcParams['svg.fonttype'] = 'none'
            # Save the figure as an SVG file
            fig.savefig(svg_path, format="svg",bbox_inches='tight')
            # Convert the SVG file to AI format
            cairosvg.svg2pdf(url=str(svg_path), write_to=str(ai_path))
            plt.close()

            ####################################################################################################################
            fig,ax = plt.subplots(figsize = (0.75*(grouping_dict[group_metric]+2),4))
            sns.stripplot(data=temp_df,x='group',y='%Carrier',hue='inheritance',hue_order=['Maternal','Paternal'],
                          palette=[colors[0],colors[1]], size=4.5)
            sns.pointplot(data=temp_df,x='group',y='%Carrier',errorbar='sd',capsize=.2,
                        linestyle="none",color='black',linewidth=1.5,estimator=np.median,
                        marker="_", markersize=20, markeredgewidth=1.5,)
            plt.axvline(x=1.5, color='black', linestyle='--')
            ax.set_xlabel('Group')
            ax.set_ylabel('Percent Carrier Inherited Telomeres')
            ax.set_title(f'Allele Enrichment ({mean_metric.capitalize()} Rank Ordered)')
            file_prefix = f'{mean_metric}_allele_enrichment_{group_metric}_inherit'
            fig.savefig(save_filepath/f'{file_prefix}.png',bbox_inches='tight')
            fig.savefig(save_filepath/f'{file_prefix}_transparent.png',bbox_inches='tight',transparent=True)
            svg_path = save_filepath/f'{file_prefix}.svg'
            ai_path = save_filepath/f'{file_prefix}.ai'
            plt.rcParams['svg.fonttype'] = 'none'
            # Save the figure as an SVG file
            fig.savefig(svg_path, format="svg",bbox_inches='tight')
            # Convert the SVG file to AI format
            cairosvg.svg2pdf(url=str(svg_path), write_to=str(ai_path))
            plt.close()

            fig,ax = plt.subplots(figsize = (0.75*(grouping_dict[group_metric]+2),4))
            sns.stripplot(data=temp_df,x='group',y='Normalized%Carrier',hue='inheritance',hue_order=['Maternal','Paternal'],
                          palette=[colors[0],colors[1]], size=4.5)
            sns.pointplot(data=temp_df,x='group',y='Normalized%Carrier',errorbar='sd',capsize=.2,
                        linestyle="none",color='black',linewidth=1.5,estimator=np.median,
                        marker="_", markersize=20, markeredgewidth=1.5,)
            plt.axvline(x=1.5, color='black', linestyle='--')
            ax.set_xlabel('Group')
            ax.set_ylabel('Normalized %Carrier Inherited Telomeres')
            ax.set_title(f'Allele Enrichment ({mean_metric.capitalize()} Rank Ordered)')
            file_prefix = f'{mean_metric}_allele_enrichment_{group_metric}_inherit_normalized'
            fig.savefig(save_filepath/f'{file_prefix}.png',bbox_inches='tight')
            fig.savefig(save_filepath/f'{file_prefix}_transparent.png',bbox_inches='tight',transparent=True)
            svg_path = save_filepath/f'{file_prefix}.svg'
            ai_path = save_filepath/f'{file_prefix}.ai'
            plt.rcParams['svg.fonttype'] = 'none'
            # Save the figure as an SVG file
            fig.savefig(svg_path, format="svg",bbox_inches='tight')
            # Convert the SVG file to AI format
            cairosvg.svg2pdf(url=str(svg_path), write_to=str(ai_path))
            plt.close()

            ####################################################################################################################
            fig,ax = plt.subplots(figsize = (0.75*(grouping_dict[group_metric]+2),4))
            sns.stripplot(data=temp_df,x='group',y='%Carrier',hue='mutation_status',hue_order=['+','-'],
                          palette=[colors[0],colors[2]],size=4.5)
            sns.pointplot(data=temp_df,x='group',y='%Carrier',errorbar='sd',capsize=.2,
                        linestyle="none",color='black',linewidth=1.5,estimator=np.median,
                        marker="_", markersize=20, markeredgewidth=1.5,)
            plt.axvline(x=1.5, color='black', linestyle='--')
            ax.set_xlabel('Group')
            ax.set_ylabel('Percent Carrier Inherited Telomeres')
            ax.set_title(f'Allele Enrichment ({mean_metric.capitalize()} Rank Ordered)')
            file_prefix = f'{mean_metric}_allele_enrichment_{group_metric}_mut'
            fig.savefig(save_filepath/f'{file_prefix}.png',bbox_inches='tight')
            fig.savefig(save_filepath/f'{file_prefix}_transparent.png',bbox_inches='tight',transparent=True)
            svg_path = save_filepath/f'{file_prefix}.svg'
            ai_path = save_filepath/f'{file_prefix}.ai'
            plt.rcParams['svg.fonttype'] = 'none'
            # Save the figure as an SVG file
            fig.savefig(svg_path, format="svg",bbox_inches='tight')
            # Convert the SVG file to AI format
            cairosvg.svg2pdf(url=str(svg_path), write_to=str(ai_path))
            plt.close()

            fig,ax = plt.subplots(figsize = (0.75*(grouping_dict[group_metric]+2),4))
            sns.stripplot(data=temp_df,x='group',y='Normalized%Carrier',hue='mutation_status',hue_order=['+','-'],
                          palette=[colors[0],colors[2]],size=4.5)
            sns.pointplot(data=temp_df,x='group',y='Normalized%Carrier',errorbar='sd',capsize=.2,
                        linestyle="none",color='black',linewidth=1.5,estimator=np.median,
                        marker="_", markersize=20, markeredgewidth=1.5,)
            plt.axvline(x=1.5, color='black', linestyle='--')
            ax.set_xlabel('Group')
            ax.set_ylabel('Normalized %Carrier Inherited Telomeres')
            ax.set_title(f'Allele Enrichment ({mean_metric.capitalize()} Rank Ordered)')
            file_prefix = f'{mean_metric}_allele_enrichment_{group_metric}_mut_normalized'
            fig.savefig(save_filepath/f'{file_prefix}.png',bbox_inches='tight')
            fig.savefig(save_filepath/f'{file_prefix}_transparent.png',bbox_inches='tight',transparent=True)
            svg_path = save_filepath/f'{file_prefix}.svg'
            ai_path = save_filepath/f'{file_prefix}.ai'
            plt.rcParams['svg.fonttype'] = 'none'
            # Save the figure as an SVG file
            fig.savefig(svg_path, format="svg",bbox_inches='tight')
            # Convert the SVG file to AI format
            cairosvg.svg2pdf(url=str(svg_path), write_to=str(ai_path))
            plt.close()
    return

def get_avg_and_rank_order(dataframe,sample_id,avg_str):
    dataframe[f'{sample_id}_{avg_str}_len'] = dataframe.apply(lambda x: get_tel_len_list(x[f'{sample_id}_read_TLs'],
                                                                True,x[f'{sample_id}_tvr_len']), axis=1).apply(lambda x: find_average(x,avg_str))
    dataframe = dataframe.sort_values(by=f'{sample_id}_{avg_str}_len')
    dataframe[f'{sample_id}_rank_order'] = [x for x in range(len(dataframe))]
    return dataframe

def compare_two_siblings(dataframe,condition1,condition2,save_path,file_prefix,average_method,
                    colors,color_order,xlabel='',ylabel='',title=''):
    with open(save_path/f'{file_prefix}_regression_stats.txt', 'w') as f:
        pass

    dataframe = dataframe.dropna(subset=[condition1,condition2])

    if xlabel=='':
        xlabel = condition1.split("_")[0]
    if ylabel=='':
        ylabel = condition2.split("_")[0]
    if title=='':
        title = f'{average_method} {xlabel} vs {ylabel}'
    
    fig,ax = plt.subplots(figsize = (5,5))
    i = sns.scatterplot(data=dataframe,x=condition1,y=condition2,hue='allele_class',hue_order=color_order,palette=colors)

    line_x = [x for x in range(round(dataframe[condition1].min()),round(dataframe[condition1].max()),round((dataframe[condition1].max() - dataframe[condition1].min())/10))] + [round(dataframe[condition1].max())]

    for i in range(len(color_order)):
        tdf = dataframe[dataframe['allele_class']==color_order[i]]
        if len(tdf)==0:
            continue
        X = tdf[condition1].tolist()
        Y = tdf[condition2].tolist()

        const_x = sm.add_constant(X, prepend=False)
        model = sm.OLS(Y, const_x)
        results = model.fit()
        slope,intercept = results.params
        r_squared = results.rsquared_adj
        p_value = results.f_pvalue

        line_y = [slope*x+intercept for x in line_x]

        if p_value <0.01:
            p_value = "{:.2e}".format(p_value)
        else:
            p_value = str(round(p_value,4))

        confidence_intervals = results.conf_int(alpha=0.05)
        lower_bounds = [x * confidence_intervals[0,0] + confidence_intervals[1,0] for x in line_x]
        upper_bounds = [x * confidence_intervals[0,1] + confidence_intervals[1,1] for x in line_x]

        j = plt.plot(line_x,line_y,color=colors[i])
        k = plt.fill_between(line_x,lower_bounds,upper_bounds,color=colors[i],
                             alpha=0.1)
        with open(save_path/f'{file_prefix}_regression_stats.txt', 'a') as f:
            f.write(f'{color_order[i]} Allele Regression:\n')
            f.write(results.summary().as_text())
            f.write('\n\n'+'='*78+'\n'+'='*78+'\n\n')
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(save_path/f'{file_prefix}_scatter.png',bbox_inches='tight')
    fig.savefig(save_path/f'{file_prefix}_scatter_transparent.png',bbox_inches='tight',transparent=True)
    svg_path = save_path/f'{file_prefix}_scatter.svg'
    ai_path = save_path/f'{file_prefix}_scatter.ai'
    plt.rcParams['svg.fonttype'] = 'none'
    # Save the figure as an SVG file
    fig.savefig(svg_path, format="svg",bbox_inches='tight')
    # Convert the SVG file to AI format
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(ai_path))
    plt.close()

    return

def make_sibling_comparison_plots(dataframe,sample_key,avg_str,save_filepath,fam_id,colors,color_order):
    sibling_groups = unique(sample_key[sample_key['Patient#']==fam_id]['SiblingGroup'].dropna().tolist())
    sibling_groups = [sample_key[sample_key['SiblingGroup']==x]['Sample'].tolist() for x in sibling_groups]
    if len(sibling_groups)<1:
        print(f'No Siblings to compare for Family {fam_id}')
        return
    for sibling_group in sibling_groups:
        sibling_pairs = list(combinations(sibling_group, 2))
        for pair in sibling_pairs:
            patient1 = pair[0]
            patient2 = pair[1]
            carrier_parent = unique(sample_key[sample_key['Sample'].apply(lambda x: x in pair)]['CarrierParent'].tolist())
            if len(carrier_parent)>1:
                print(f'Samples {patient1} and {patient2} parental disagreement. No figure made. Check inputs.')
                continue
            carrier_parent = carrier_parent[0]
            non_carrier_parent = unique(sample_key[sample_key['Sample'].apply(lambda x: x in pair)]['NonCarrierParent'].tolist())
            if len(non_carrier_parent)>1:
                print(f'Samples {patient1} and {patient2} parental disagreement. No figure made. Check inputs.')
                continue
            non_carrier_parent = non_carrier_parent[0]
            rel_cols = [x for x in dataframe.columns if x.split('_')[0] in [patient1,patient2,carrier_parent,non_carrier_parent]]
            pair_df = dataframe[rel_cols].dropna(subset=[f'{patient1}_tvr_consensus',f'{patient2}_tvr_consensus']).copy()
            pair_df['allele_class'] = 'Uncertain'
            if str(carrier_parent)!='nan':
                pair_df.loc[pair_df.dropna(subset = f'{carrier_parent}_tvr_consensus').index,'allele_class']='Carrier'
            if str(non_carrier_parent)!='nan':
                pair_df.loc[pair_df.dropna(subset = f'{carrier_parent}_tvr_consensus').index,'allele_class']='NonCarrier'
            for patient in [patient1,patient2]:
                pair_df = get_avg_and_rank_order(pair_df,patient,avg_str)
            
            condition1 = f'{patient1}_rank_order'
            condition2 = f'{patient2}_rank_order'
            patient1_name = sample_key[sample_key['Sample']==patient1]['FigureName'].item()
            patient2_name = sample_key[sample_key['Sample']==patient2]['FigureName'].item()
            xlabel = f'{patient1_name} Rank Order'
            ylabel = f'{patient2_name} Rank Order'
            title = f'{patient1_name} vs {patient2_name} {avg_str} Rank Order'

            compare_two_siblings(pair_df,condition1,condition2,save_filepath,f'{patient1}_vs_{patient2}_{avg_str}_RO',
                                 avg_str,colors,color_order,xlabel,ylabel,title)
            
            condition1 = f'{patient1}_{avg_str}_len'
            condition2 = f'{patient2}_{avg_str}_len'
            xlabel = f'{patient1_name} {avg_str} Telomere Length'
            ylabel = f'{patient2_name} {avg_str} Telomere Length'
            title = f'{patient1_name} vs {patient2_name} {avg_str} Telomere Length'

            compare_two_siblings(pair_df,condition1,condition2,save_filepath,f'{patient1}_vs_{patient2}_{avg_str}_len',
                                 avg_str,colors,color_order,xlabel,ylabel,title)
    return

def main():
    ######################################## These are the editable parameters ########################################
    lev_ratio_used = 85
    analysis_dir = Path('/data/annika/Nanopore/Telogator2/POT1Analysis/SecondRoundSeq/UpdatedMergedFastq/TelogatorAnalysis/AnalysisAndFigures-Final')
    patient_key = pd.read_excel('/data/annika/Nanopore/Telogator2/POT1Analysis/SecondRoundSeq/PatientKey.xlsx')
    add_tvr_len_to_TL = True
    average_functions = ['mean','median','75pct']

    hue_order = ['Carrier','NonCarrier','Uncertain']
    custom_colors = ['firebrick','steelblue','darkgrey']

    stacked_bar_colors_dict = {'C': 'lightgrey','A': 'darkolivegreen','D': 'gold',
                            'E': 'seagreen','F': 'indianred','G': 'navy',
                            'H': 'yellowgreen','I': 'royalblue','K': 'darkseagreen',
                            'L': 'cornflowerblue','M': 'deepskyblue','N': 'goldenrod',
                            'P': 'darkorange','Q': 'steelblue','R': 'rebeccapurple',
                            'S': 'mediumvioletred','T': 'darkmagenta','V': 'teal',
                            'W': 'firebrick','Y': 'lightseagreen'}
    ########################################################################################################################

    aggregated_df = pd.read_csv(analysis_dir/f'3_Collapsed_Cluster_Correlation_L{lev_ratio_used}.tsv',delimiter='\t')
    base_columns = ['chromosome','allele_id','allele_ref_tvr','allele_ref_dataset','mean_lev_ratio']

    RO_dir = analysis_dir/'Rank_Ordered_Barplots'
    if not os.path.isdir(RO_dir):
        os.mkdir(RO_dir)
    
    sibling_dir = analysis_dir/'Sibling_Comparisons'
    if not os.path.isdir(sibling_dir):
        os.mkdir(sibling_dir)
    
    patient_stack_dir = analysis_dir/'Patient_Stacked_Bar'
    if not os.path.isdir(patient_stack_dir):
        os.mkdir(patient_stack_dir)
    
    allele_stack_dir = analysis_dir/'Allele_Stacked_Bar'
    if not os.path.isdir(allele_stack_dir):
        os.mkdir(allele_stack_dir)

    proband_comp_dir = analysis_dir/'Proband_Sample_Comp'
    if not os.path.isdir(proband_comp_dir):
        os.mkdir(proband_comp_dir)

    parental_comp_dir = analysis_dir/'Parental_Comp'
    if not os.path.isdir(parental_comp_dir):
        os.mkdir(parental_comp_dir)

    std_dev_dir = analysis_dir/'Standard_Deviation'
    if not os.path.isdir(std_dev_dir):
        os.mkdir(std_dev_dir)

    enrichment_dir = analysis_dir/'Allele_Enrichment'
    if not os.path.isdir(enrichment_dir):
        os.mkdir(enrichment_dir)

    enrichment_values = []
    for patient_num in unique(patient_key['Patient#'].tolist()): 
        family_samples = patient_key[patient_key['Patient#']==patient_num]['Sample'].tolist()
        family_columns = [x for x in aggregated_df.columns if x.split('_')[0] in family_samples]
        family_df = aggregated_df[base_columns+family_columns].copy()

        #filter out any rows which have alleles not relating to this family
        tvr_len_cols = [x for x in family_df.columns if '_tvr_len' in x]
        family_df['patients_per_cluster'] = family_df[tvr_len_cols].apply(lambda x: x>0).sum(axis=1)
        family_df = family_df[family_df.patients_per_cluster>0].copy().reset_index(drop=True)

        #define the sample that is supposed to set the allele order for the family
        if len(family_samples)>1:
            allele_order_sample = [x for x in family_samples if patient_key[patient_key.Sample==x].SetOrdering.item()==True][0]
        else:
            allele_order_sample = family_samples[0]
        
        for operator_for_average in average_functions:
            RO_figure_filepath = RO_dir/operator_for_average
            if not os.path.isdir(RO_figure_filepath):
                os.mkdir(RO_figure_filepath)

            family_df = rename_allele_ids_by_length(family_df,add_tvr_len_to_TL,allele_order_sample,operator_for_average)
            family_df.to_csv(RO_figure_filepath/f'3.{patient_num}_{operator_for_average}_Family_Clusters_Renamed.tsv',
                            sep='\t',index=False)
            proband_samples = {}
            for patient_sample in family_samples:
                patient_figurename = patient_key[patient_key['Sample']==patient_sample]['FigureName'].item()
                print(f'Generating {operator_for_average} plots for {patient_figurename}... {datetime.now()}')
                
                patient_cols = [x for x in family_columns if patient_sample in x]+['plot_allele_id']
                patient_df = family_df[patient_cols].copy()
                patient_df = patient_df.dropna(subset=f'{patient_sample}_tvr_len')

                patient_df['total_len'] = patient_df.apply(lambda x: 
                                                        get_tel_len_list(read_string=x[f'{patient_sample}_read_TLs'],
                                                                            adjust=add_tvr_len_to_TL,
                                                                            tvr_len=x[f'{patient_sample}_tvr_len']),axis=1)
                patient_df[f'{operator_for_average}_len'] = patient_df['total_len'].apply(lambda x: find_average(x,operator_for_average))
                patient_df = patient_df.sort_values(by=f'{operator_for_average}_len').reset_index(drop=True)

                #classify alleles as coming from either the carrier, non-carrier, or Uncertain parent
                #check to see if there are parents associated with the sample
                carrier_parent_sample = patient_key[patient_key['Sample']==patient_sample]['CarrierParent'].item()
                noncarrier_parent_sample = patient_key[patient_key['Sample']==patient_sample]['NonCarrierParent'].item()
                    
                #if there are parents associated with the samples, match the allele ids to determine source
                patient_df = get_carrier_classification(patient_df,family_df,carrier_parent_sample,noncarrier_parent_sample)
                patient_df['order'] = [x for x in range(len(patient_df))]
                patient_df[f'parental_{operator_for_average}_len']= patient_df.apply(lambda x: get_parental_tl(x,family_df,
                                                                            carrier_parent_sample,noncarrier_parent_sample,
                                                                            operator_for_average,
                                                                            add_tvr_len_to_TL),axis=1)
                patient_df['std_dev'] = patient_df['total_len'].apply(lambda x: np.std(x))
                # compile the proband samples which are sequenced twice
                if 'Proband' in patient_figurename:
                    proband_samples[patient_sample] = patient_df

                #make standard deviation scatter plots
                make_scatter_plots(f'{operator_for_average}_len','std_dev',patient_df,custom_colors,hue_order,
                                            std_dev_dir,f'{operator_for_average.capitalize()} Cluster Telomere Length',
                                            'Standard Deviation',f'{patient_figurename} Standard Deviation vs {operator_for_average.capitalize()}',
                                            f'{patient_sample}_{operator_for_average}_std_dev',True,xmin=0,xmax=16000,ymin=0,ymax=6000)
                make_rank_ordered_barplot(patient_df,patient_figurename,operator_for_average,hue_order,custom_colors,
                                        RO_figure_filepath,overlay_parental=False)
                make_patient_specific_stacked_bar_chart(patient_df,stacked_bar_colors_dict,patient_figurename,custom_colors,
                                                        patient_sample,patient_stack_dir,operator_for_average)
                
                #Only make the following plots if there is parental information for a given sample
                if len([x for x in patient_df[f'parental_{operator_for_average}_len'].tolist() if x!=None])<=0:
                    continue
                # Rank order the parental alleles
                patient_df = patient_df.sort_values(by=f'parental_{operator_for_average}_len')
                total_parental = len(patient_df.dropna(subset=f'parental_{operator_for_average}_len'))
                patient_df['parental_rank_order'] = [x for x in range(total_parental)] + [np.nan]*(len(patient_df)-total_parental)
                patient_df = patient_df.sort_index()
                patient_df.loc[patient_df.dropna(subset=f'parental_{operator_for_average}_len').index,'inherited_rank_order'] = [x for x in range(total_parental)]
                parental_comp_scatter_plot(patient_df,operator_for_average,custom_colors,hue_order,parental_comp_dir,
                                           patient_sample,patient_figurename,True)
                make_rank_ordered_barplot(patient_df,patient_figurename,operator_for_average,hue_order,custom_colors,
                                            RO_figure_filepath,overlay_parental=True)

                #make quartile and thirds plots for each patient, collect the information for final plot
                patient_skew = get_patient_inheritence_skew(patient_df,patient_sample,patient_figurename,enrichment_dir,custom_colors,
                                                            hue_order,operator_for_average,patient_key)
                patient_skew['method'] = operator_for_average
                enrichment_values.append(patient_skew)
            #make stacked bar plots for shared alleles in families
            if len(family_samples)<=1:
                continue

            family_bar_savepath = allele_stack_dir/f'Family{patient_num}'
            if not os.path.isdir(family_bar_savepath):
                os.mkdir(family_bar_savepath)
            
            majority_alleles = family_df[family_df['patients_per_cluster']>len(tvr_len_cols)/2]['plot_allele_id'].tolist()
            make_allele_stacked_bar_charts(majority_alleles,family_df,family_samples,family_columns,operator_for_average,
                                           patient_key,stacked_bar_colors_dict,patient_num,family_bar_savepath,add_tvr_len_to_TL)
            # do sibling comparisons
            make_sibling_comparison_plots(family_df,patient_key,operator_for_average,sibling_dir,patient_num,custom_colors,hue_order)
            if len(proband_samples)<=1:
                continue
            get_proband_sample_info_for_comp(proband_samples,operator_for_average,proband_comp_dir,patient_num)
    
    #make total enrichment plots for skew
    enrichment_values = pd.concat(enrichment_values)
    enrichment_values[['Normalized%Carrier','Normalized%Other']] = enrichment_values.apply(lambda x: normalize_to_non_subsetted(x,enrichment_values),axis=1).tolist()
    enrichment_values.to_csv(enrichment_dir/'parental_proportion_dataframe.csv',index=False)
    make_skew_summary_plots(enrichment_values,enrichment_dir,custom_colors,hue_order)
            

if __name__ == '__main__':
    main()