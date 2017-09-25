#!/usr/bin/env python
import pandas as pd
import pandas.io.sql as psql
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
#import datetime
#import matplotlib.dates as mdates
#from pandas.core import datetools
#from pandas.core import datetools
import pandas.core.tools.datetimes
import warnings
warnings.filterwarnings('ignore')
import sklearn
import statsmodels
from collections import defaultdict
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import itertools
import sys
import os 

def main():
    company_id = int(sys.argv[1])
#    print company_id
    #os.system("rm results*.csv")
    all_orders_over_time = pd.read_csv("updated_all_orders.csv")
    all_orders_over_time['order_date'] = pd.to_datetime(all_orders_over_time['order_date'], errors='coerce').dt.date
    all_orders_over_time[all_orders_over_time.isnull().any(axis=1)].sort_values(["order_date"], ascending=False).reset_index(drop=True)
    all_orders_over_time.shape
    all_orders_over_time.head(2)

    description_dict = dict(zip(all_orders_over_time.variant_id,all_orders_over_time.prod_description))


    ##get date range
    earliestDate =  all_orders_over_time["order_date"].min()
    latestDate =  all_orders_over_time["order_date"].max()
    #print earliestDate
    #print latestDate



    ##--- get all orders by company--- 
    def get_all_orders_by_company(df, company_id):
        comp_order_df = df[df["company_id"] == company_id].sort_values('order_date', ascending=False).reset_index(drop=True)
        return comp_order_df 


    # In[86]:

    ##--- get last order by company--- 
    def get_last_order_by_company(df, company_id):
        comp_order_df = get_all_orders_by_company(df, company_id)
        last_order_num = comp_order_df.at[0,'actual_order_id']
        last_order_df = comp_order_df[comp_order_df["actual_order_id"] == last_order_num]
        return last_order_df 


    # In[87]:

    def get_last_order_items(df, company_id):
        last_order_df = get_last_order_by_company(df, company_id)
        list_items_in_last_order = last_order_df.variant_id.unique()
        return list_items_in_last_order


    # In[88]:

    def get_all_order_items(df, company_id):
        comp_order_df = get_all_orders_by_company(df, company_id)
        list_items_in_all_order = comp_order_df.variant_id.unique()
        return list_items_in_all_order


    # In[89]:

    def get_non_last_order_items(df, company_id):
        list_items_in_all_order = get_all_order_items(df, company_id)
        list_items_in_last_order = get_last_order_items(df, company_id)
        list_prev_purch_items_not_in_last_order = list(set(list_items_in_all_order) - set(list_items_in_last_order))
        return list_prev_purch_items_not_in_last_order


    # In[90]:

    def get_all_except_last(df, company_id):
        all_order_df = get_all_orders_by_company(df, company_id)
        last_order_df = get_last_order_by_company(df, company_id)
        i1 = all_order_df.index
        i2 = last_order_df.index
        all_minus_last_df = all_order_df[~i1.isin(i2)]
        return all_minus_last_df
            


    # In[91]:

    def getTopFiveItemsNotInLastOrder(df, company_id):
        all_minus_last_df = get_all_except_last(df, company_id)
        all_minus_last_df["count"] = all_minus_last_df['variant_id'].groupby(all_minus_last_df['variant_id']).transform('count')
        all_minus_last_df["avg_quantity"] = all_minus_last_df['quantity'].groupby(all_minus_last_df['variant_id']).transform('mean')
        all_minus_last_df = all_minus_last_df.groupby(all_minus_last_df['variant_id']).first()
        all_minus_last_df.avg_quantity = all_minus_last_df.avg_quantity.astype(int)
        all_minus_last_df = all_minus_last_df.sort_values("count", ascending = False).reset_index(drop=True)
        if (all_minus_last_df.shape[0] >= 5):
            return all_minus_last_df[["variant_id", "prod_description", "avg_quantity"]].head(5)
        else: 
            return all_minus_last_df[["variant_id","prod_description", "avg_quantity"]]


    # In[92]:

    def encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1


    # In[93]:

    def getAssociations(df):
        min_support_threshold = 0.07
        lift_threshold = 1.1
        confidence_threshold = 0.7

        basket = (df.groupby(['actual_order_id', "variant_id"])['quantity']
                  .sum().unstack().reset_index().fillna(0)
                  .set_index('actual_order_id'))
        #basket.head()
        basket_sets = basket.applymap(encode_units)
        #basket_sets.head()
        frequent_itemsets = apriori(basket_sets, min_support=min_support_threshold, use_colnames=True)
        frequent_itemsets.head()
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        #print rules.shape
        rules = rules[ (rules['lift'] >= lift_threshold) & (rules['confidence'] >= confidence_threshold) ]
        #print rules.shape
        rules = rules.sort_values(["support"], ascending = False)
        #print rules.head()
        return rules


    # In[94]:

    #convert first 2 columns to string
    #keep only single item antecedants/ remove double or more combinations 
    def format_rules(df):
        df[["antecedants", "consequents"]] = df[["antecedants", "consequents"]].astype(str) 
        df = df[~df["antecedants"].str.contains(',')]
        df.antecedants = df.antecedants.str.extract('(\d+)')
        df.consequents = df.consequents.str.extract('(\d+)') 
        df[["antecedants", "consequents"]] = df[["antecedants", "consequents"]].astype(int) 
        #print df
        return df


    # In[95]:

    def getProductDescriptors(df, filtered_on_order):
        df = df.drop_duplicates(subset='variant_id', keep='first', inplace=False) 
        df1 = pd.merge(filtered_on_order[["antecedants", "consequents"]], df[["variant_id", "prod_description"]], how = 'inner', left_on = 'antecedants', right_on = 'variant_id')
        df2 = df1.rename(columns = {"variant_id": "Product ID", "prod_description": "Product Description"})
        df3 = pd.merge(df2, df[["variant_id", "prod_description"]], how = 'inner', left_on = 'consequents', right_on = 'variant_id')
        df4 = df3.rename(columns = {"variant_id": "Product ID", "prod_description": "Product Description"})
        df5 = df4.drop(["antecedants", "consequents"], axis = 1)
        return df5


    # In[96]:

    def getCommonBoughtTogether(df, list_items):
        rules = getAssociations(all_orders_over_time)
        rules_df = format_rules(rules)
        filtered_on_order = rules_df[rules_df['antecedants'].isin(list_items)]
        #print filtered_on_order.head()
        filtered_on_order = filtered_on_order.drop_duplicates(subset='antecedants', keep='first', inplace=False)
        #print filtered_on_order.head()
        detailed_df = getProductDescriptors(df, filtered_on_order)
        if (detailed_df.shape[0] >= 5):
            #print "yes"
            return detailed_df.head(5)
        else: 
            #print "no"
            return detailed_df


    # In[97]:

    
    

    if (all_orders_over_time[all_orders_over_time["company_id"] == company_id].shape[0] == 0):

        return 0
    else:    
        all_order_df = get_all_orders_by_company(all_orders_over_time, company_id)
        #print all_order_df.shape

        last_order_df = get_last_order_by_company(all_orders_over_time, company_id)
        #print last_order_df.shape
        #print last_order_df.head(2)

        list_items_in_all_order = get_all_order_items(all_orders_over_time, company_id)
        #print len(list_items_in_all_order)

        list_items_in_last_order = get_last_order_items(all_orders_over_time, company_id)
        #print len(list_items_in_last_order)

        list_prev_purch_items_not_in_last_order = get_non_last_order_items(all_orders_over_time, company_id)
        #print len(list_prev_purch_items_not_in_last_order)

        all_minus_last_df = get_all_except_last(all_order_df, company_id)
        #print all_minus_last_df.shape

        topFiveItemsNotInLastOrder = getTopFiveItemsNotInLastOrder(all_orders_over_time, company_id)
        #print topFiveItemsNotInLastOrder


    # In[98]:

    def getLastWeightedAvg(curr_timeseries):
        decay_halflife = 4
        expwighted_avg = pd.ewma(curr_timeseries, halflife=decay_halflife)
        return int(expwighted_avg.iloc[-1])


    # In[119]:

    def createTimeSeries(x, y):
        new_index = pd.to_datetime(x) 
        select_item = pd.DataFrame(index=new_index)
        select_item["Quantity"] = y
        ts = select_item["Quantity"].squeeze()
        return ts


    # In[120]:

    basket = (all_order_df.sort_values('order_date', ascending = True).groupby(['order_date', "variant_id"])['quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('order_date'))
    #drop any columns/rows that are all zeros 
    basket = basket[(basket.T != 0).any()]
    basket = basket.loc[:, (basket != 0).any(axis=0)]
    basket.index = pd.to_datetime(basket.index)
    basket.head(2)

    rolling_avg_quantity_list = []
    for var_id in list_items_in_last_order:
        x = basket.index
        y = basket[var_id]
        ts = createTimeSeries(x, y)
        lastWeightedAvg = getLastWeightedAvg(ts)
        rolling_avg_quantity_list.append(lastWeightedAvg)

      


    # In[108]:

    filteron = ["variant_id", "prod_description", "quantity"]
    #print "Recommended order: "
    recom_order = last_order_df.filter(items = filteron)
    se = pd.Series(rolling_avg_quantity_list)
    recom_order["quantity"] =  se.values
    recom_order = recom_order[recom_order['quantity'] > 0]
    recom_order_items = recom_order["variant_id"]
    recom_order_display = recom_order.copy(deep = True)
    recom_order_display.rename(columns = {"variant_id": "Product ID", "prod_description": "Product Description","quantity": "Quantity"}, inplace = True)
    #print recom_order_display
    #print 
    #print 
    last_order = last_order_df.filter(items = filteron)
    last_order.rename(columns = {"variant_id": "Product ID", "prod_description": "Product Description","quantity": "Quantity"}, inplace = True)
    #print "Last order: "
    #print last_order
    #print 
    #print "Commonly ordered by this company:"
    top_items = topFiveItemsNotInLastOrder
    top_items.rename(columns = {"variant_id": "Product ID", "prod_description": "Product Description","avg_quantity": "Quantity"}, inplace = True)
    #print top_items
    #print 
    #print 
    #print "Frequently purchased together:"
    list_items = recom_order_items.tolist()
    boughtTogetherPairs = getCommonBoughtTogether(all_orders_over_time, list_items)
    #top_items.rename(columns = {"variant_id": "Product ID", "prod_description": "Product Description","avg_quantity": "Quantity"}, inplace = True)
    #print boughtTogetherPairs
    recom_order_display.to_csv("results-order.csv")
    last_order.to_csv("results-last.csv")
    top_items.to_csv("results-often.csv")
    boughtTogetherPairs.to_csv("results-together.csv")
    return 1

if __name__ == '__main__':
    main()




