import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import pandas.io.sql as psql
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.dates as mdates
import datetime as dt
from datetime import datetime
import time
from datetime import timedelta
from collections import defaultdict, Counter
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from fbprophet import Prophet 
from itertools import chain, combinations
import math
import matplotlib.ticker as ticker
import psycopg2
import sys

all_orders_over_time = pd.read_csv("new_updated_all_orders.csv")
all_orders_over_time['order_date'] = pd.to_datetime(all_orders_over_time['order_date'], errors='coerce').dt.date
all_orders_over_time["variant_id"] = all_orders_over_time["variant_id"].astype(int)
all_orders_over_time.head(2)
description_dict = dict(zip(all_orders_over_time.variant_id,all_orders_over_time.prod_description))

##--- get all orders by company--- 
def get_all_orders_by_company(df, company_id):
    comp_order_df = df[df["company_id"] == company_id].sort_values('order_date', ascending=False).reset_index(drop=True)
    return comp_order_df 

##--- get last order by company--- 
def get_last_order_by_company(df, company_id):
    comp_order_df = get_all_orders_by_company(df, company_id)
    last_order_num = comp_order_df.at[0,'actual_order_id']
    last_order_df = comp_order_df[comp_order_df["actual_order_id"] == last_order_num]
    return last_order_df

##--- get last 10% of order by company--- 
def get_last_ten_percent_order_by_company(df, company_id):
    comp_order_df = get_all_orders_by_company(df, company_id)
    uniqueOrders = comp_order_df["actual_order_id"].drop_duplicates(keep = "first", inplace = False)
    numOrders = uniqueOrders.shape[0]
    tenPercentOrders = int(numOrders/10)
    last_order_num_list = uniqueOrders.head(tenPercentOrders).tolist()
    last_order_df = comp_order_df[comp_order_df["actual_order_id"].isin(last_order_num_list)]
    return last_order_df 

##--- get list of last order placed by a company---
def get_last_order_items(df, company_id):
    last_order_df = get_last_order_by_company(df, company_id)
    list_items_in_last_order = last_order_df.variant_id.unique()
    return list_items_in_last_order

##--- get last order date by a company---
def get_last_order_date(df, company_id):
    comp_order_df = get_all_orders_by_company(df, company_id)
    last_order_num = comp_order_df.at[0,'actual_order_id']
    last_order_df = comp_order_df[comp_order_df["actual_order_id"] == last_order_num]
    last_order_date = str(last_order_df["order_date"][0])
    return last_order_date

##--- get all items ordered by a company---
def get_all_order_items(df, company_id):
    comp_order_df = get_all_orders_by_company(df, company_id)
    list_items_in_all_order = comp_order_df.variant_id.unique()
    return list_items_in_all_order

##--- get all items ordered by a company not in last order---
def get_non_last_order_items(df, company_id):
    list_items_in_all_order = get_all_order_items(df, company_id)
    list_items_in_last_order = get_last_order_items(df, company_id)
    list_prev_purch_items_not_in_last_order = list(set(list_items_in_all_order) - set(list_items_in_last_order))
    return list_prev_purch_items_not_in_last_order

##--- get all orders by a company except for last order---
def get_all_except_last(df, company_id):
    all_order_df = get_all_orders_by_company(df, company_id)
    last_order_df = get_last_order_by_company(df, company_id)
    i1 = all_order_df.index
    i2 = last_order_df.index
    all_minus_last_df = all_order_df[~i1.isin(i2)]
    return all_minus_last_df
        
##--- get all top products purchased by company not in last order---
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

##--- get all top products purchased by company not in current forecast---
def getTopFiveItemsNotInCurrentOrder(df, trim_basket):
    all_order_df = df
    groupbysize = all_order_df.groupby("variant_id").size().sort_values(ascending = False)
    groupMean = all_order_df.groupby("variant_id")["quantity"].mean().astype(int)
    avgQuantity = groupMean.to_dict()
    filteredgroups = groupbysize.drop(groupbysize[trim_basket].index)
    toplist = filteredgroups.index.tolist()
    filteron = ["variant_id", "prod_description", "quantity"]
    filtered_all_order = all_order_df.filter(items = filteron)
    new_order = filtered_all_order[filtered_all_order['variant_id'].isin(toplist)]
    filtered_new_order = new_order.drop_duplicates(subset='variant_id', keep='first', inplace=False).reset_index(drop=True)
    filtered_new_order['Avg_quantity'] = filtered_new_order['variant_id'].map(avgQuantity)
    filtered_new_order = filtered_new_order.drop("quantity", axis=1)
    if (filtered_new_order.shape[0] >= 5):
        return filtered_new_order.head(5)
    else: 
        return filtered_new_order

##---one hot encoding for recommendation system 
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

##---calculate similiarity of groups 
def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)

##---get rules for associating groups 
def getAssociations(df):
    ###----these values have not been optimized-------------------###
    ###----parameters should be adjusted for desired stringency---###
    min_support_threshold = 0.07
    lift_threshold = 1.1
    confidence_threshold = 0.7

    basket = (df.groupby(['actual_order_id', "variant_id"])['quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('actual_order_id'))
    basket_sets = basket.applymap(encode_units)
    frequent_itemsets = apriori(basket_sets, min_support=min_support_threshold, use_colnames=True)
    frequent_itemsets.head()
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules = rules[ (rules['lift'] >= lift_threshold) & (rules['confidence'] >= confidence_threshold) ]
    rules = rules.sort_values(["support"], ascending = False)
    return rules

##---create all possible groupings of products 
def filter_rules(rules, list_items):
    all_combo = []
    for L in range(0, len(list_items)+1):
          for subset in itertools.combinations(list_items, L):
            all_combo.append(frozenset(subset))
    all_combo.pop(0) #remove empty set 

    all_index = []
    for index, row in rules.iterrows():
        for i in all_combo:
            if (row["antecedants"] == (i)):
                all_index.append(index)
                next
    filtered_index_list = list(set(all_index))
    filtered_df = rules.loc[filtered_index_list]
    filtered_df['cos_similiarity'] = filtered_df['antecedants'].map(lambda a: counter_cosine_similarity(Counter(set(list_items)),Counter(a)))
    sorted_df = filtered_df.sort_values( "cos_similiarity",ascending=False).reset_index(drop=True)  
    return sorted_df

##---format similar groups into list 
def getProductDescriptors(filtered_on_order):
    for index, row in filtered_on_order.iterrows():
        prod_list = list(row["antecedants"])
        desc_list = []
        prod_list2 = list(row["consequents"])
        desc_list2 = []
        for p in prod_list:
            var_id = int(p)
            str1 = description_dict[var_id]
            str2 = str1.replace(",", " ")
            desc_list.append(str2)     
        for p in prod_list2:
            var_id = int(p)
            str1 = description_dict[var_id]
            str2 = str1.replace(",", " ")
            desc_list2.append(str2)  
        desc_list.append("\n")
        desc_list2.append("\n")    
        str1 = ','.join(desc_list)
        str2 = ','.join(desc_list2)
        filtered_on_order.set_value(index, 'current_list', str1)
        filtered_on_order.set_value(index, 'recommend_list', str2)
        filtered_on_order.set_value(index, 'current_list_size', len(desc_list))
        filtered_on_order.set_value(index, 'recommend_list_size', len(desc_list2))
    filtered_on_order.sort_values( ["current_list_size", "recommend_list_size"],ascending=[False, False], inplace = True)
    filtered_on_order["current_list"] = filtered_on_order["current_list"].astype(str)
    filtered_on_order["recommend_list"] = filtered_on_order["recommend_list"].astype(str)
    sorted_filtered_on_order = filtered_on_order.drop_duplicates(subset="current_list", keep='first', inplace=False)
    sorted_filtered_on_order = filtered_on_order.drop_duplicates(subset="recommend_list", keep='first', inplace=False)
    return sorted_filtered_on_order

##--function for getting groups of frequently purchased items 
def getCommonBoughtTogether(df, list_items):
    rules = getAssociations(df)
    filtered_on_order = filter_rules(rules, list_items)
    filtered_on_order["current_list"] = filtered_on_order["antecedants"]
    filtered_on_order["recommend_list"] = filtered_on_order["antecedants"]
    filtered_on_order["current_list_size"] = 0
    filtered_on_order["recommend_list_size"] = 0
    detailed_df = getProductDescriptors(filtered_on_order)
    if (detailed_df.shape[0] >= 5):
        return detailed_df.head(5)
    else: 
        return detailed_df

def createTimeSeries(x, y):
    new_index = pd.to_datetime(x) 
    select_item = pd.DataFrame(index=new_index)
    select_item["Quantity"] = y
    ts = select_item["Quantity"].squeeze()
    return ts

def getCompanyGraph (fig, label, ax, order_date, quantity,earliestDate, latestDate): 
    #ax = fig.add_subplot(numRow, numCol, counter)
    plt.xticks(rotation=70)
    ax.plot(order_date, quantity, linestyle="dashed", marker = "o", markerfacecolor="None", markeredgecolor="red", markeredgewidth=2, label=label)
    ax.set_xlim(earliestDate, latestDate)
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=70 )
    ax.legend()
    return


def getProphet(df):
    #ProphetModel 
    m = Prophet()
    m.fit(df);
    #add today to dataframe
    future_df = df.filter(["ds"], axis=1)
    today_df = pd.DataFrame(data = [dt.date.today()], columns = ["ds"])
    future = future_df.append(today_df, ignore_index=True)
    forecast = m.predict(future)
    forecast["yhat"] = forecast["yhat"].astype(int)
    return forecast

def getProphetPredictionGraph(var_id, label, ax, forecast_df, observedsize, xminaxis, xmaxaxis, ymin, ymax):
    ##forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    x = forecast["ds"]
    x = pd.to_datetime(x)
    y = forecast["yhat"]
    xobs = x[0:observedsize]
    yobs = y[0:observedsize]
    xpred =  x[observedsize-1:]
    ypred =  y[observedsize-1:]
    xpredp =  x[observedsize:]
    ypredp =  y[observedsize:]
    plt.xticks(rotation=80, ha="right")
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.plot(xobs,yobs, linestyle="-", marker = "o", markerfacecolor="None", markeredgecolor="blue", markeredgewidth=1, label="Observed")
    ax.plot(xpred, ypred, linestyle="--", color = "red", label="_nolegend_")
    ax.plot(xpredp,ypredp, linestyle="--", markersize=8,  marker = "o", color = "red", label='Predicted')
    p = ax.plot(xpredp,ypredp,  marker = "o", markersize=14, markerfacecolor="None", markeredgecolor="red", markeredgewidth=1,color = "red", label='_nolegend_')
    ax.set_title(label)
    ax.set_autoscale_on(False)
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=80 )
    ax.set_xlim(xmin = xmin, xmax = xmax)
    yint = range(ymin, math.ceil(ymax)+3)
    ax.yaxis.set_ticks(yint)
    ax.legend(loc=2)
    return 


def createPlot(trim_basket): 
    numItems = len(trim_basket)
    numCol = 2
    numRow = numItems // numCol + (numItems % numCol > 0)
    reminder = numItems%numCol
    height = 4 * numRow

    #fig, axes = plt.subplots(nrows=numRow, ncols=numCol, sharex=True, sharey=True, figsize=(12, height))
    fig, axes = plt.subplots(nrows=numRow, ncols=numCol, figsize=(12, height))
    #axes_list = [item for sublist in axes for item in sublist] 

    axes_list = axes.tolist()

    counter = 0
    for var_id in trim_basket:
        label =  description_dict[var_id]
        forecast_df = df_trim[var_id]   
        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
        #predictionsize = forecast_df.shape[0]-df.shape[0]
        observedsize = forecast_df.shape[0] - 1
        ax = fig.axes[counter]
        getProphetPredictionGraph(var_id, label, ax, forecast_df, observedsize, xminaxis, xmaxaxis, ymin, ymax)
        counter = counter + 1
        
    while counter < numItems:
        ax = (fig.axes[counter])
        ax.plot(0,0)
        counter = counter +1
        #ax.tick_params(labelbottom='off')  

    fig.text(0.5, -0.01, 'Order date', ha='center')
    fig.text(-0.01, 0.5, 'Quantity', va='center', rotation='vertical')
    plt.suptitle("Company: "+str(company_id), y=1.01)

    fig.tight_layout()
    fig.show()
    return fig 