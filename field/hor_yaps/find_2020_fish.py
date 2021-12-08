
# Find periods when fish tracks might be present:
df=pd.read_csv('yaps/full/2020/20200317T0600-20200520T0000/all_detections.csv')
grp=df.groupby('tag')

tag_counts=pd.DataFrame( {'count':grp.size(),
                          't_start':grp['epo'].min(),
                          't_stop':grp['epo'].max() })

tag_counts=tag_counts.sort_values(by=['count'],ascending=False)

tag_counts=tag_counts[ tag_counts['count']>10 ]

##

pd.set_option('display.max_rows',2500)
fish_tag_counts=tag_counts.filter(axis=0,regex='^[^F][^F]..$')

fish_tag_counts=fish_tag_counts.copy()
fish_tag_counts['duration']=fish_tag_counts['t_stop']-fish_tag_counts['t_start']
epo_mid=0.5*(fish_tag_counts['t_stop']+fish_tag_counts['t_start'])
fish_tag_counts['date']=utils.unix_to_dt64(epo_mid.values)
