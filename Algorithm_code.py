from datetime import timedelta, datetime
import statsmodels.api as sm
import numpy as np
import pandas as pd
from Selection.FundamentalUniverseSelectionModel import FundamentalUniverseSelectionModel
from sklearn.decomposition import PCA

class SMAPairsTrading(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2000, 1 , 1 )   
        self.SetEndDate(2002, 1 , 1 )
        self.SetCash(100000)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.Universe.Index.QC500)
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Raw
        self.AddAlpha(PairsTradingAlphaModel())
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(TrailingStopRiskManagementModel(0.03))
        self.SetBenchmark("SPY")
        self.SetSecurityInitializer(self.CustomSecurityInitializer)
        self.buy = pd.DataFrame()
        self.sell = pd.DataFrame()
        self.liquidate = pd.DataFrame()
        
        
    def OnEndOfDay(self, symbol):
        self.Log("Taking a position of " + str(self.Portfolio[symbol].Quantity) + " units of symbol " + str(symbol))
        
    def CustomSecurityInitializer(self, security):
        security.SetLeverage(1)

class PairsTradingAlphaModel(AlphaModel):

    def __init__(self):
        self.pair = []
        self.period = timedelta(days=1)
        
    def Update(self, algorithm, data):
        
        List=[x.Symbol for x in self.pair]
        history = algorithm.History(List, 61 ).close.unstack(level=0)
        self.buy,self.sell,self.liquidate = self.GetIndexes( history)
            
        Appd = []
        
        for i in self.buy:
            Appd.append(Insight.Price(i,self.period, InsightDirection.Up,None,None,None))#,None, None, None,0.02))
                        
        for i in self.sell:
            Appd.append(Insight.Price(i,self.period, InsightDirection.Down,None,None,None))
                        
        for i in self.liquidate:
            Appd.append(Insight.Price(i,self.period, InsightDirection.Flat,None,None,None))
              
        return Insight.Group([ x for x in Appd])
        
            
    def GetIndexes(self, history):
       
        # Sample data for PCA 
        sample = history.dropna(axis=1).pct_change().dropna()
        
        sample_mean = sample.mean()
        
        sample_std = sample.std()
        
        sample = ((sample-sample_mean)/(sample_std)) #Normalizing 

        # Fit the PCA model for sample data
        model = PCA().fit(sample)
        
        #Distributing eigenportfolios 
        
        EigenPortfolio = pd.DataFrame(model.components_)
        
        EigenPortfolio.columns = sample.columns
        
        EigenPortfolio = EigenPortfolio/sample_std
        
        EigenPortfolio = ( EigenPortfolio.T / EigenPortfolio.sum(axis=1) )

        # Get the first n_components factors
        factors = np.dot(sample, EigenPortfolio)[:,:1]  # we want to replicate the market 
        

        # Add 1's to fit the linear regression (intercept)
        factors = sm.add_constant(factors)

        # Train Ordinary Least Squares linear model for each stock
        OLSmodels = {ticker: sm.OLS(sample[ticker], factors).fit() for ticker in sample.columns}

        # Get the residuals from the linear regression after PCA for each stock
        resids = pd.DataFrame({ticker: model.resid for ticker, model in OLSmodels.items()})

        # Get the OU parameters 
        shifted_residuals = resids.cumsum().iloc[1:,:]
        
        resids = resids.cumsum().iloc[:-1,:]
        
        resids.index = shifted_residuals.index
        
        OLSmodels2 = {ticker: sm.OLS(resids[ticker],sm.add_constant(shifted_residuals[ticker])).fit() for ticker in resids.columns} 
        
        # Get the new residuals
        
        resids2 = pd.DataFrame({ticker: model.resid for ticker, model in OLSmodels2.items()})
        
        # Get the mean reversion parameters 
        
        a = pd.DataFrame({ticker : model.params[0] for ticker , model in OLSmodels2.items()},index=["a"])
        
        b = pd.DataFrame({ticker: model.params[1] for ticker , model in OLSmodels2.items()},index=["a"])
        
        e = (resids2.std())/(252**(-1/2))
        
        k = -np.log(b) * 252
        
        #Get the z-score
        
        var = (e**2 /(2 * k) )*(1 - np.exp(-2 * k * 252))
        
        num = -a * np.sqrt(1 - b**2)
        
        den = ( 1-b ) * np.sqrt( var )
        
        m  = ( a / ( 1 - b ) )
        
        zscores=(num / den ).iloc[0,:]# zscores of the most recent day
        
        # Get the stocks far from mean (for mean reversion)
        
        selected_buy = zscores[zscores < -1.25].dropna().sort_values()[:20]
        
        selected_sell = zscores[zscores > 1.25].dropna().sort_values()[-20:]
        
        selected_liquidate = zscores[abs(zscores) < 0.50 ]

        # Return each selected stock
        weights_buy = selected_buy.index
        
        weights_sell = selected_sell.index
        
        weights_liquidate = selected_liquidate.index
        
        return weights_buy, weights_sell, weights_liquidate
    
    def OnSecuritiesChanged(self, algorithm, changes):
        self.pair = [x for x in changes.AddedSecurities]
