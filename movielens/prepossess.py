import pandas as pd
import numpy as np
item_genre=pd.read_csv('u.item',sep='|',encoding='unicode_escape')
total_data=pd.read_csv('u.data',sep='\t',header=None)
comedy=item_genre.iloc[:,0][(item_genre.iloc[:,10]==1)]
romance=item_genre.iloc[:,0][(item_genre.iloc[:,19]==1)]
drama=item_genre.iloc[:,0][(item_genre.iloc[:,13]==1)]
action=item_genre.iloc[:,0][(item_genre.iloc[:,6]==1)]
thriller=item_genre.iloc[:,0][(item_genre.iloc[:,21]==1)]
data={}
genres=['comedy','romance','drama','action','thriller']
for genre in genres:
    exec('index=%s.copy()'%genre)
    data[genre]=total_data.loc[total_data[1].isin(index),[0,1,2]]
    exec('user_%s=set(data[genre][0].unique())'%genre)
exec('user_share=user_%s.intersection(user_%s,user_%s,user_%s,user_%s)'%tuple(genres))
for genre,item in data.items():
    data_sh=item[item[0].isin(user_share)].copy()
    for i in range(2):
        a=item[i].unique()
        b=data_sh[i].unique()
        a.sort()
        b.sort()
        for ii,jj in enumerate(a):
            item.loc[item[i]==jj,i]=ii+1
        for ii,jj in enumerate(b):
            data_sh.loc[data_sh[i]==jj,i]=ii+1
    data[genre]=item
    data_sh.to_csv('%s_share.csv'%genre,header=None,index=False)
total_data=total_data[[0,1,2]]
total_data.to_csv('total.csv',header=None,index=False)
