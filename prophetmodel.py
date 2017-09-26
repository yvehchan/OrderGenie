#!usr/bin/env python

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import pandas.io.sql as psql
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.dates as mdates
#from pandas.core import datetools
import datetime as dt
from datetime import datetime
import time
from datetime import timedelta
#import sklearn
#import statsmodels
from collections import defaultdict
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from fbprophet import Prophet 


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


def get_last_order_items(df, company_id):
    last_order_df = get_last_order_by_company(df, company_id)
    list_items_in_last_order = last_order_df.variant_id.unique()
    return list_items_in_last_order


def get_all_order_items(df, company_id):
    comp_order_df = get_all_orders_by_company(df, company_id)
    list_items_in_all_order = comp_order_df.variant_id.unique()
    return list_items_in_all_order


# In[11]:

def get_non_last_order_items(df, company_id):
    list_items_in_all_order = get_all_order_items(df, company_id)
    list_items_in_last_order = get_last_order_items(df, company_id)
    list_prev_purch_items_not_in_last_order = list(set(list_items_in_all_order) - set(list_items_in_last_order))
    return list_prev_purch_items_not_in_last_order


# In[12]:

def get_all_except_last(df, company_id):
    all_order_df = get_all_orders_by_company(df, company_id)
    last_order_df = get_last_order_by_company(df, company_id)
    i1 = all_order_df.index
    i2 = last_order_df.index
    all_minus_last_df = all_order_df[~i1.isin(i2)]
    return all_minus_last_df
        


# In[13]:

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


# In[14]:

def getTopFiveItemsNotInCurrentOrder(df, company_id):
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


# In[15]:

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


# In[24]:

def getAssociations(df):
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


# In[25]:

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


# In[26]:

def getProductDescriptors(df, filtered_on_order):
    df = df.drop_duplicates(subset='variant_id', keep='first', inplace=False) 
    df1 = pd.merge(filtered_on_order[["antecedants", "consequents"]], df[["variant_id", "prod_description"]], how = 'inner', left_on = 'antecedants', right_on = 'variant_id')
    df2 = df1.rename(columns = {"variant_id": "Product ID", "prod_description": "Product Description"})
    df3 = pd.merge(df2, df[["variant_id", "prod_description"]], how = 'inner', left_on = 'consequents', right_on = 'variant_id')
    df4 = df3.rename(columns = {"variant_id": "Product ID", "prod_description": "Product Description"})
    df5 = df4.drop(["antecedants", "consequents"], axis = 1)
    return df5


# In[27]:

def getCommonBoughtTogether(df, list_items):
    rules = getAssociations(all_orders_over_time)
    rules_df = format_rules(rules)
    filtered_on_order = rules_df[rules_df['antecedants'].isin(list_items)]
    #print filtered_on_order.head()
    filtered_on_order = filtered_on_order.drop_duplicates(subset='antecedants', keep='first', inplace=False)
    #print filtered_on_order.head()
    detailed_df = getProductDescriptors(df, filtered_on_order)
    if (detailed_df.shape[0] >= 5):
        return detailed_df.head(5)
    else: 
        return detailed_df


# In[28]:

def getLastWeightedAvg(curr_timeseries):
    decay_halflife = 4
    expwighted_avg = pd.ewma(curr_timeseries, halflife=decay_halflife)
    return int(expwighted_avg.iloc[-1])


# In[29]:

def createTimeSeries(x, y):
    new_index = pd.to_datetime(x) 
    select_item = pd.DataFrame(index=new_index)
    select_item["Quantity"] = y
    ts = select_item["Quantity"].squeeze()
    return ts


# In[30]:

def getCompanyGraph (fig, label, ax, order_date, quantity,earliestDate, latestDate): 
    #ax = fig.add_subplot(numRow, numCol, counter)
    plt.xticks(rotation=70)
    ax.plot(order_date,quantity, linestyle="dashed", marker = "o", markerfacecolor="None", markeredgecolor="red", markeredgewidth=2, label=label)
    ax.set_xlim(earliestDate, latestDate)
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=70 )
    ax.legend()
    return

     
def getWeightedAvgGraph(fig, var_id, label, ax, order_date, quantity,earliestDate, latestDate):
    moving_avg_window = 8
    decay_halflife = 4
    ts = createTimeSeries(order_date, quantity)
    expwighted_avg = pd.ewma(ts, halflife=decay_halflife)
    #print expwighted_avg
    #ts_ewma_diff = ts - expwighted_avg
    plt.xticks(rotation=70)
    ax.plot(ts, label = "Observed", markersize=3, marker = "o", markerfacecolor="None", markeredgecolor="blue", markeredgewidth=1)
    ax.plot(expwighted_avg, color='red', linestyle="dashed", label = "Exp. rolling mean")
    ax.set_title(label)
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

def getProphetPredictionGraph(fig, var_id, label, ax, forecast, observedsize, xmin, xmax):
    ##forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    x = forecast["ds"]
    x = pd.to_datetime(x)
    y = forecast["yhat"]
    y_lower = forecast["yhat_lower"]
    y_upper = forecast["yhat_upper"]
    xobs = x[0:observedsize]
    yobs = y[0:observedsize]
    xpred =  x[observedsize-1:]
    ypred =  y[observedsize-1:]
    xpredp =  x[observedsize:]
    ypredp =  y[observedsize:]
    ypred_lower =  y_lower[observedsize-1:]
    ypred_upper =  y_upper[observedsize-1:]
    #print (len(xobs), len(yobs), len(x), len(y), len(xpred), len(ypred))
    plt.xticks(rotation=70)
    #ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.fill_between(xpred.values, ypred_lower, ypred_upper, facecolor='red', alpha = 0.1, interpolate=True, label = "Est. error")
    ax.plot(xpred,ypred, linestyle="--", color = "red", label="Prediction")
    ax.plot(xpredp,ypredp,  marker = "o",color = "red", label='_nolegend_')
    ax.plot(xobs,yobs, linestyle="-", marker = "o", markerfacecolor="None", markeredgecolor="blue", markeredgewidth=1, label="Observed")
    ax.set_title(label)
    ax.set_xlim(xmin, xmax)
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=70 )
    ax.legend(loc=2)
    return 

def main():
    #company_id = int(sys.argv[1])
    #list of dataframes 
    results_dfs = []

    all_orders_over_time = pd.read_csv("updated_all_orders.csv")
    all_orders_over_time['order_date'] = pd.to_datetime(all_orders_over_time['order_date'], errors='coerce').dt.date
    all_orders_over_time[all_orders_over_time.isnull().any(axis=1)].sort_values(["order_date"], ascending=False).reset_index(drop=True)
    description_dict = dict(zip(all_orders_over_time.variant_id,all_orders_over_time.prod_description))

    ##get date range
    earliestDate =  all_orders_over_time["order_date"].min()
    xmin = earliestDate
    xmax = dt.date.today()

    if (all_orders_over_time[all_orders_over_time["company_id"] == company_id].shape[0] == 0):
        #returns empty list
        return results_dfs
    else:    
        all_order_df = get_all_orders_by_company(all_orders_over_time, company_id)
        last_order_df = get_last_order_by_company(all_orders_over_time, company_id)
        list_items_in_all_order = get_all_order_items(all_orders_over_time, company_id)
        list_items_in_last_order = get_last_order_items(all_orders_over_time, company_id)
        list_prev_purch_items_not_in_last_order = get_non_last_order_items(all_orders_over_time, company_id)
        all_minus_last_df = get_all_except_last(all_order_df, company_id)
        #topFiveItemsNotInLastOrder = getTopFiveItemsNotInLastOrder(all_orders_over_time, company_id)

        basket = (all_order_df.sort_values('order_date', ascending = True).groupby(['order_date', "variant_id"])['quantity']
                  .sum().unstack().reset_index().fillna(0)
                  .set_index('order_date'))
        basket = basket[basket.columns[(basket==0).sum(axis=0) > 10]]
        basket = basket[(basket.T != 0).any()]
        basket = basket.loc[:, (basket != 0).any(axis=0)]
        basket.index = pd.to_datetime(basket.index)


        trim_basket = [] 
        df_trim = {}
        var_quantit = {}

        for var_id in basket:
            s1 = pd.Series(basket.index, name = "ds").reset_index(drop = True)
            s2 = pd.Series(basket[var_id], name = "y").reset_index(drop = True)
            df = pd.concat([s1, s2], axis=1)
            df["ds"] = pd.to_datetime(df["ds"])
            forecast = getProphet(df)
            forecast_quant = forecast["yhat"].iloc[-1]
            #print (forecast_quantity)
            if (forecast_quant > 0):
                trim_basket.append(var_id)
                df_trim[var_id] = forecast


        numItems = len(trim_basket)
        numCol = 2
        numRow = numItems // numCol + (numItems % numCol > 0)
        reminder = numItems%numCol
        height = 4 * numRow


        #fig, axes = plt.subplots(nrows=numRow, ncols=numCol, sharex=True, sharey=True, figsize=(12, height))
        #axes_list = axes.tolist()

        #for var_id in trim_basket:
        #    label =  description_dict[var_id]
        #    forecast_df = df_trim[var_id]
        #    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
        #    predictionsize = forecast_df.shape[0]-df.shape[0]
        #    observedsize = forecast_df.shape[0] - predictionsize
        #    ax = axes_list.pop(0)
        #    getProphetPredictionGraph(fig, var_id, label, ax, forecast_df, observedsize, xmin, xmax)

        #while axes_list:
        #    ax = axes_list.pop(0)
        #    ax.plot(0,0)
        #    ax.tick_params(labelbottom='off')  

        #fig.text(0.5, -0.01, 'Order date', ha='center')
        #fig.text(-0.01, 0.5, 'Quantity', va='center', rotation='vertical')
        #plt.suptitle("Company: "+str(company_id), y=1.01)

        #fig.tight_layout()
        #fig.show()


    # In[40]:

        filteron = ["variant_id", "prod_description", "quantity"]
        #print "Recommended order: "
        filtered_all_order = all_order_df.filter(items = filteron)
        new_order = filtered_all_order[filtered_all_order['variant_id'].isin(trim_basket)]
        filtered_new_order = new_order.drop_duplicates(subset='variant_id', keep='first', inplace=False).reset_index(drop=True)
        for var_id in trim_basket:
            filtered_new_order['quantity'][filtered_new_order.variant_id == var_id] = int(var_quantit[var_id])


        recom_order_items = filtered_new_order["variant_id"]
        recom_order_display = filtered_new_order.copy(deep = True)
        recom_order_display.rename(columns = {"variant_id": "Product ID", "prod_description": "Product Description","quantity": "Quantity"}, inplace = True)

        last_order = last_order_df.filter(items = filteron)
        last_order.rename(columns = {"variant_id": "Product ID", "prod_description": "Product Description","quantity": "Quantity"}, inplace = True)

        top_items = getTopFiveItemsNotInCurrentOrder(all_order_df, trim_basket)
        top_items.rename(columns = {"variant_id": "Product ID", "prod_description": "Product Description","Avg_quantity": "Avg quantity"}, inplace = True)


        list_items = recom_order_items.tolist()
        boughtTogetherPairs = getCommonBoughtTogether(all_orders_over_time, list_items)
        boughtTogetherPairs.rename(columns = {"variant_id": "Product ID", "prod_description": "Product Description","avg_quantity": "Quantity"}, inplace = True)

        results_dfs.append(recom_order_display)
        results_dfs.append(last_order)
        results_dfs.append(top_items)
        results_dfs.append(boughtTogetherPairs)
        return results_dfs


if __name__ == '__main__':
    main()



