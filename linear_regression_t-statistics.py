#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def get_table(X,Y):
    lm = LinearRegression(fit_intercept = True)
    lm.fit(X,Y)
   
    params = np.append(lm.intercept_,lm.coef_)
    newX = np.append(np.ones((len(X),1)), X, axis=1)
    
    predictions = lm.predict(X)
    MSE = (np.sum((Y-predictions)**2))/(len(newX)-len(newX[0]))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]

    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["p-value"] = [params,sd_b,ts_b,p_values]
    print(myDF3)

