from flask import Flask, render_template, url_for, redirect, request
import os
import subprocess
from subprocess import Popen, PIPE
import pandas as pd
import prophetmodel

app = Flask(__name__)

@app.route("/")
def main():
    return render_template("index.html")

@app.route('/forecast', methods=['GET','POST'])
def forecast():
  company = request.args.get('company')
  #p = Popen(["Prophet.py", str(company)], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
  #myVars = {"company_id":company}
  #query_results = exec(open('prophetmodel.py').read(), myVars)
  query_results = prophetmodel.main(company)
 # query_results=os.system("python freshStocker.py "+company)
  if (query_results.shape[0] == 0):
        return render_template("index.html")  
  else:    
    #query_results is a list of dfs
    recom_order = query_results[0]
    last_order = query_results[1]
    top_order = query_results[2]
    together_order = query_results[3]
    return render_template("output.html", comp= company, tables=[data1.to_html(classes='Order'), data2.to_html(classes='Last'), 
      data3.to_html(classes='Often'), data4.to_html(classes='Together')], 
      titles = ['na', 'Recommended order', 'Last order', "Often ordered", "Frequently ordered together"])

@app.route('/stat')
def stat():
  company = request.args.get('company')
  p = Popen(["freshStocker.py", str(company)], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
  query_results=p.communicate()[0]
 # query_results=os.system("python freshStocker.py "+company)
  if query_results == 0:
    return render_template("inputerror.html")  
  else:    
    data1 = pd.read_csv('results-order.csv', index_col=0).reset_index(drop=True)
    data2 = pd.read_csv('results-last.csv', index_col=0).reset_index(drop=True)
    data3 = pd.read_csv('results-often.csv', index_col=0).reset_index(drop=True)
    data4 = pd.read_csv('results-together.csv', index_col=0).reset_index(drop=True) 
    return render_template("output.html", comp= company, tables=[data1.to_html(classes='Order'), data2.to_html(classes='Last'), 
      data3.to_html(classes='Often'), data4.to_html(classes='Together')], 
      titles = ['na', 'Recommended order', 'Last order', "Often ordered", "Frequently ordered together"])



if __name__ == "__main__":
    app.run(debug=True)
