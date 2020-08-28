import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

class PcaStatArbitrageAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2001, 1, 1)       # Set Start Date
        self.SetEndDate(2001, 5, 10)         # Set End Date
        self.SetCash(100000)                # Set Strategy Cash

        self.nextRebalance = self.Time      # Initialize next rebalance time
        self.rebalance_days = 0            # Rebalance every 30 days

        self.lookback = 61                  # Length(days) of historical data
        self.num_components = 15             # Number of principal components in PCA
        self.num_equities = 500               # Number of the equities pool
        self.weights_buy = pd.DataFrame()       # Pandas data frame (index: symbol) that stores the weight
        self.weights_sell = pd.DataFrame()
        self.weights_liquidate = pd.DataFrame()

        self.UniverseSettings.Resolution = Resolution.Daily   # Use hour resolution for speed
        self.AddUniverse(self.CoarseSelectionAndPCA)         # Coarse selection + PCA
        #self.AddRiskManagement(MaximumDrawdownPercentPerSecurity(0.03))
        


    def CoarseSelectionAndPCA(self, coarse):
        '''Drop securities which have too low prices.
        Select those with highest by dollar volume.
        Finally do PCA and get the selected trading symbols.
        '''

        # Before next rebalance time, just remain the current universe
        #if self.Time < self.nextRebalance:
        #    return Universe.Unchanged

        ### Simple coarse selection first

        # Sort the equities in DollarVolume decendingly
        selected = sorted([x for x in coarse if x.Price > 5],
                          key=lambda x: x.DollarVolume, reverse=True)

        symbols = [x.Symbol for x in selected[:self.num_equities]]

        ### After coarse selection, we do PCA and linear regression to get our selected symbols

        # Get historical data of the selected symbols
        history = self.History(symbols, self.lookback, Resolution.Daily).close.unstack(level=0)

        # Select the desired symbols and their weights for the portfolio from the coarse-selected symbols
        try:
            self.weights_buy,self.weights_sell,self.weights_liquidate = self.GetWeights(history)
        except:
            self.weights_buy,self.weights_sell,self.weights_liquidate = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
            
        # If there is no final selected symbols, return the unchanged universe
        if self.weights_buy.empty or self.weights_sell.empty or self.weights_liquidate.empty:
            return Universe.Unchanged

        return [x for x in symbols if str(x) in self.weights_buy.index or str(x) in self.weights_sell.index or str(x) in self.weights_liquidate]


    def GetWeights(self, history):
        '''
        Get the finalized selected symbols and their weights according to their level of deviation
        of the residuals from the linear regression after PCA for each symbol
        '''
        # Sample data for PCA 
        sample = history.dropna(axis=1).pct_change().dropna()
        
        sample_mean = sample.mean() 
        
        sample_std = sample.std()
        
        sample = ((sample-sample_mean)/(sample_std)) * 252 **(1/2) # Center it column-wise

        # Fit the PCA model for sample data
        model = PCA().fit(sample)
        
        #Distributing eigenportfolios 
        
        EigenPortfolio = pd.DataFrame(model.components_)
        
        EigenPortfolio.columns = sample.columns
        
        EigenPortfolio = EigenPortfolio/sample_std
        
        EigenPortfolio = ( EigenPortfolio.T / EigenPortfolio.sum(axis=1) )

        # Get the first n_components factors
        factors = np.dot(sample, EigenPortfolio)[:,:self.num_components]
        
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
        
        e = resids2.std() * 252 **( 1 / 2)
        
        k = -np.log(b) * 252
    
        #Get the z-score
        
        var = (e**2 /(2 * k) )*(1 - np.exp(-2 * k * 252))
        
        num = -a * np.sqrt(1 - b**2)
        
        den =( ( 1-b ) * np.sqrt( var )).dropna(axis=1)
        
        m  = ( a / ( 1 - b ) ).dropna(axis=1)
        
        zscores=(num / den ).iloc[0,:]# zscores of the most recent day
        
        # Get the stocks far from mean (for mean reversion)
        
        selected_buy = zscores[zscores < -1.5]
        
        selected_sell = zscores[zscores > 1.5]
        
        selected_liquidate = zscores[abs(zscores) < 0.50 ]
        
        #summing all orders
        
        sum_orders = selected_buy.abs().sum() + selected_sell.abs().sum()

        # Return the weights for each selected stock
        weights_buy = selected_buy * (1 / sum_orders)
        
        weights_sell = selected_sell * (1 / sum_orders)
        
        weights_liquidate = selected_liquidate
        
        return weights_buy.sort_values(),weights_sell.sort_values(),weights_liquidate.sort_values()


    def OnData(self, data):
        '''
        Rebalance every self.rebalance_days
        '''
        ### Do nothing until next rebalance
        #if self.Time < self.nextRebalance:
        #    return

        ### Open positions
        for symbol, weight in self.weights_buy.items():
            # If the residual is way deviated from 0, we enter the position in the opposite way (mean reversion)
            if self.Securities[symbol].Invested:
                continue
            self.SetHoldings(symbol, -weight)
            
        ### short positions
        for symbol, weight in self.weights_sell.items():
            if self.Securities[symbol].Invested:
                continue
            self.SetHoldings(symbol,-weight)
            
        for symbol, weight in self.weights_liquidate.items():
            self.Liquidate(symbol)
            
        ### Update next rebalance time
        #self.nextRebalance = self.Time + timedelta(self.rebalance_days)


    def OnSecuritiesChanged(self, changes):
        '''
        Liquidate when the symbols are not in the universe
        '''
        for security in changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol, 'Removed from Universe')
        #    self.SetHoldings("SPY", 1)
