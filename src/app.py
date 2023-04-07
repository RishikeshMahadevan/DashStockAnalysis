import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from yahoo_fin.stock_info import *

app = dash.Dash(__name__)
server=app.server

def fundamental(string):
    string=string.upper()
    
    import requests
    import pandas as pd
    import requests
    import pandas as pd
    import yfinance as yf
    ticker=string.upper()
    
    def financialratios(string):
       
        # Set API key and base URL
        ticker=string.upper()
        API_KEY = '90c1c9da624fcab78500a10f9634b0cf'
        BASE_URL = 'https://financialmodelingprep.com/api/v3'

        # Define function to get financial statements data and convert it to pandas dataframe
        def get_financial_statements(ticker, statement_type):
            # Define API endpoint and parameters
            endpoint = f'{BASE_URL}/{statement_type}/{ticker}'
            params = {'limit': 120, 'apikey': API_KEY}

            # Make API request and get JSON response
            response = requests.get(endpoint, params=params)
            data = response.json()

            # Convert data to pandas dataframe
            df = pd.DataFrame(data)
            #df = df.set_index('date')

            return df

        # Define ticker and statement types

        statement_types = ['income-statement', 'balance-sheet-statement', 'cash-flow-statement']

        # Loop through statement types and get financial statements dataframes
        financial_data = {}
        for statement_type in statement_types:
            df = get_financial_statements(ticker, statement_type)
            financial_data[statement_type] = df
        income_statement_df=financial_data['income-statement']
        balance_sheet_df=financial_data['balance-sheet-statement']
        cash_flow_df=financial_data['cash-flow-statement']





        # Replace "None" with "NaN" in the DataFrames
        income_statement_df = income_statement_df.replace("None", "NaN")
        balance_sheet_df = balance_sheet_df.replace("None", "NaN")
        cash_flow_df = cash_flow_df.replace("None", "NaN")



        # Calculate the ratios
        try:
            current_ratio = balance_sheet_df['totalCurrentAssets'][0] / balance_sheet_df['totalCurrentLiabilities'][0]
        except (ZeroDivisionError, KeyError, TypeError):
            current_ratio = "N/A"
        try:
            quick_ratio = (balance_sheet_df['totalCurrentAssets'][0] - balance_sheet_df['inventory'][0]) / balance_sheet_df['totalCurrentLiabilities'][0]
        except (ZeroDivisionError, KeyError, TypeError):
            quick_ratio = "N/A"
        try:
            roe = income_statement_df['netIncome'][0] / balance_sheet_df['totalStockholdersEquity'][0]
        except (ZeroDivisionError, KeyError, TypeError):
            roe = "N/A"
        try:
            roa = income_statement_df['netIncome'][0] / balance_sheet_df['totalAssets'][0]
        except (ZeroDivisionError, KeyError, TypeError):
            roa = "N/A"
        try:
            roi = income_statement_df['netIncome'][0] / ((balance_sheet_df['totalAssets'][0] + balance_sheet_df['totalAssets'][0]) / 2)
        except (ZeroDivisionError, KeyError, TypeError):
            roi = "N/A"
        try:
            gross_profit_margin = income_statement_df['grossProfit'][0] / income_statement_df['revenue'][0]
        except (ZeroDivisionError, KeyError, TypeError):
            gross_profit_margin = "N/A"
        try:
            operating_profit_margin = income_statement_df['operatingIncome'][0] / income_statement_df['revenue'][0]
        except (ZeroDivisionError, KeyError, TypeError):
            operating_profit_margin = "N/A"
        try:
            net_profit_margin = income_statement_df['netIncome'][0] / income_statement_df['revenue'][0]
        except (ZeroDivisionError, KeyError, TypeError):
            net_profit_margin = "N/A"
        try:
            eps = income_statement_df['eps'][0]
        except (KeyError, TypeError):
            eps = "N/A"
        try:
            ticker = yf.Ticker(ticker)
            latest_price = ticker.history(period="1d")['Close'][0]
            pe_ratio = latest_price/eps
        except (KeyError, TypeError):
            pe_ratio = "N/A"

        try:
            total_revenue = income_statement_df['revenue'][0]
        except (KeyError, TypeError):
            total_revenue = "N/A"
        try:
            cost_of_goods_sold = income_statement_df['costOfRevenue'][0]
        except (KeyError, TypeError):
            cost_of_goods_sold = "N/A"
        try:
            accounts_receivable = (balance_sheet_df['netReceivables'][0] + balance_sheet_df['netReceivables'][1]) / 2
        except (KeyError, TypeError):
            accounts_receivable = "N/A"
        try:
            inventory = (balance_sheet_df['inventory'][0] + balance_sheet_df['inventory'][1]) / 2
        except (KeyError, TypeError):
            inventory = "N/A"
        try:
            accounts_payable = (balance_sheet_df['accountPayables'][0] + balance_sheet_df['accountPayables'][1]) / 2
        except (KeyError, TypeError):
            accounts_payable = "N/A"

        # Calculate the ratios
        try:
            dso = accounts_receivable / (total_revenue / 365)
        except (KeyError, TypeError, ZeroDivisionError):
            dso = "N/A"

        try:
            dio = inventory / (cost_of_goods_sold / 365)
        except (KeyError, TypeError, ZeroDivisionError):
            dio = "N/A"

        try:
            dpo = accounts_payable / (cost_of_goods_sold / 365)
        except (KeyError, TypeError, ZeroDivisionError):
            dpo = "N/A"
        try:
            ccc = dso + dio - dpo
        except (KeyError, TypeError, ZeroDivisionError):
            ccc = "N/A"


        total_debt = balance_sheet_df['totalDebt'][0]
        total_equity=balance_sheet_df['totalStockholdersEquity'][0]
        total_assets=balance_sheet_df['totalAssets'][0]

        try:
            debt_to_equity = total_debt / total_equity
        except (KeyError, TypeError, ZeroDivisionError):
            debt_to_equity = "N/A"

        try:
            debt_to_capital = total_debt / (total_debt + total_equity)
        except (KeyError, TypeError, ZeroDivisionError):
            debt_to_capital = "N/A"

        try:
            debt_to_assets = total_debt / total_assets
        except (KeyError, TypeError, ZeroDivisionError):
            debt_to_assets = "N/A"

        try:
            financial_leverage = total_assets / total_equity
        except (KeyError, TypeError, ZeroDivisionError):
            financial_leverage = "N/A"

        try:
            debt_to_ebitda = total_debt / income_statement_df['ebitda'][0]
        except (KeyError, TypeError, ZeroDivisionError):
            debt_to_ebitda = "N/A"

        try:
            ebit = income_statement_df['incomeBeforeTax'][0]
            interest_expense = income_statement_df['interestExpense'][0]
            interest_coverage = ebit / interest_expense
        except (KeyError, TypeError, ZeroDivisionError):
            interest_coverage = "N/A"



        # Convert the ratios to a DataFrame with column names
        ratios_df = pd.DataFrame({
            "Current Ratio": [current_ratio],
            "Quick Ratio": [quick_ratio],
            "Return on Equity (ROE)": [roe],
            "Return on Assets (ROA)": [roa],
            "Gross Profit Margin": [gross_profit_margin],
            "Operating Profit Margin": [operating_profit_margin],
            "Net Profit Margin": [net_profit_margin],
            "Earnings Per Share (EPS)": [eps],
            "Price to Earnings (P/E) Ratio": [pe_ratio],
            "Days Sales Outstanding (DSO)": [dso],
            "Days Inventory Outstanding (DIO)": [dio],
            "Cash Conversion Cycle (CCC)": [ccc],
            "Debt to Equity Ratio": [debt_to_equity],
            "Debt to Capital Ratio": [debt_to_capital],
            "Debt to Assets Ratio": [debt_to_assets],
            "Financial Leverage Ratio":[financial_leverage],
            "Debt to EBITDA Ratio":[debt_to_ebitda],
            "Interest Coverage": [interest_coverage]


        })
        ratios_df=ratios_df.transpose()
        ratios_df=ratios_df.rename(columns={0: string})
        ratios_df['Ratios']=ratios_df.index
        ratios_df=ratios_df.reindex(columns=['Ratios',string])
        #ratios_df=ratios_df.rename(columns={0:income_statement_df['fiscalDateEnding'].iloc[0]})
        #ratios_df.set_index(income_statement_df['fiscalDateEnding'].iloc[0])
        return ratios_df
    def reportfile(sec):
                  
        if sec=='Industrials':
            df=pd.read_csv('https://raw.githubusercontent.com/RishikeshMahadevan/Fundamental-stock-analyser/main/Industrials.csv')
            #return df
        elif sec=='Health Care':
            df=pd.read_csv('https://raw.githubusercontent.com/RishikeshMahadevan/Fundamental-stock-analyser/main/Healthcare.csv')
            #return df
        elif sec=='Information Technology':
            df=pd.read_csv('https://raw.githubusercontent.com/RishikeshMahadevan/Fundamental-stock-analyser/main/Informationtechnology.csv')
            #return df
        elif sec=='Communication Services':
            df=pd.read_csv('https://raw.githubusercontent.com/RishikeshMahadevan/Fundamental-stock-analyser/main/Communicationservices.csv')
            #return df
        elif sec=='Consumer Staples':
            df=pd.read_csv('https://raw.githubusercontent.com/RishikeshMahadevan/Fundamental-stock-analyser/main/Consumerstaples.csv')
            #return df
        elif sec=='Consumer Discretionary':
            df=pd.read_csv('https://raw.githubusercontent.com/RishikeshMahadevan/Fundamental-stock-analyser/main/Consumerdiscretionary.csv')
            #return df
        elif sec=='Utilities':
            df=pd.read_csv('https://raw.githubusercontent.com/RishikeshMahadevan/Fundamental-stock-analyser/main/Utilities.csv')
            #return df
        elif sec=='Materials':
            df=pd.read_csv('https://raw.githubusercontent.com/RishikeshMahadevan/Fundamental-stock-analyser/main/Materials.csv')
            #return df
        elif sec=='Real Estate':
            df=pd.read_csv('https://raw.githubusercontent.com/RishikeshMahadevan/Fundamental-stock-analyser/main/Realestate.csv')
            #return df
        elif sec=='Energy':
            df=pd.read_csv('https://github.com/RishikeshMahadevan/Fundamental-stock-analyser/blob/main/Energy.csv')
            #return df
        elif sec=='Financials':
            p=1

            return 0
        else:
            return 0


        l=df.columns.to_list()
        l=l[1::]
        pd.options.display.float_format = '{:.3f}'.format
        df=df.fillna("Not Available")
        #df=df.iloc[::,-1]
        df.rename(columns={'Unnamed: 0':'Financial Ratios'},inplace=True)
        df=df.set_index(df['Financial Ratios'])
        #df=df.drop(l,axis=1)
        df.rename(columns={'0':'Sector Average'},inplace=True)
        df=df.drop(['Financial Ratios'],axis=1)
        #df=df.T
        #df=df.T

        return df
    sector=pd.read_csv('https://raw.githubusercontent.com/RishikeshMahadevan/SectorAnalyzer/main/sp500.csv')
    tick=sector['Symbol'].to_list()
    sect=sector['GICS Sector'].to_list()
    stringfinderindex=0
    for i in range(0,len(tick)-1):
        if ticker==tick[i]:
            stringfinderindex=i
    sec=sect[stringfinderindex]
    def sectorfile(ticker,sec):
  
        if sec=='Financials':
            def finfinancialratiocalc(string):
                pd.options.display.precision = 3
                

                # Create Ticker object for JPMorgan
                jp = get_stats(string)
                jp=jp[31:]

                return jp


            df=finfinancialratiocalc(ticker)
            return df
        else:
            pd.options.display.float_format
            df=reportfile(sec)
            return df
    if sec=="Financials":
        fr=sectorfile(string,sec)

    else:
        fr=pd.concat([financialratios(string),sectorfile(string,sec)],axis=1)
       
  
    return fr


#app layout################################################





app.layout = html.Div([
    html.H1("Stock Analysis Dashboard"),
    html.Div([
        html.Label("Enter a stock ticker:"),
        dcc.Input(
            id="ticker-input",
            type="text",
            value="AAPL",
            style={"marginRight": "10px"}
        ),
        html.Button(id="submit-button", n_clicks=0, children="Submit")
    ], style={"marginBottom": "25px"}),
    dcc.Tabs(id="tabs-example", value='tab-1', children=[
        dcc.Tab(label='Technical Analysis', value='tab-1', children=[
            html.Label("Select a time period:"),
            dcc.Dropdown(
                id="timeframe-dropdown",
                options=[
                    {'label': '1 minute', 'value': '1m'},
                    {'label': '30 minutes', 'value': '30m'},
                    {'label': '1 hour', 'value': '1h'},
                    {'label': '3 hour', 'value': '3h'},
                    {'label': '1 day', 'value': '1d'},
                    {'label': '1 month', 'value': '1mo'},
                ],
                value='1d',
                style={"width": "150px", "marginRight": "10px"} 
            ),
            html.Div([
                html.Label('Moving Averages'),
                dcc.Checklist(
                    id='moving-averages',
                    options=[
                        {'label': '10 T', 'value': 10},
                        {'label': '25 T', 'value': 25},
                        {'label': '50 T', 'value': 50},
                    ],
                    value=[10, 25, 50]
                ),
            ]),
            html.Div([
                html.Label('Bollinger Bands'),
                dcc.Checklist(
                    id='bollinger-bands',
                    options=[
                        {'label': 'Show Bollinger Bands', 'value': 'show'},
                    ],
                    value=[]
                ),
                html.Label('Standard Deviation'),
                dcc.Slider(
                    id='std-dev-slider',
                    min=1,
                    max=5,
                    step=0.1,
                    value=2
                )
            ]),
            html.Div([
                html.Label('RSI'),
                dcc.Checklist(
                    id='rsi',
                    options=[
                        {'label': 'Show RSI', 'value': 'show'},
                    ],
                    value=[]
                )
            ]),
            html.Div([
                html.Label('MACD'),
                dcc.Checklist(
                    id='macd',
                    options=[
                        {'label': 'Show MACD', 'value': 'show'},
                    ],
                    value=[]
                    )
                ]),

        dcc.Graph(id='stock-chart')
        ]),
        
        dcc.Tab(label='Fundamentals', value='tab-2', children=[
            html.Table(id='stock-data')
        ]),
        
    ]),
])

@app.callback(
    [Output('stock-chart', 'figure'),
     Output('stock-data', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('ticker-input', 'value'),
    State('timeframe-dropdown', 'value'),
    State('moving-averages', 'value'),
    State('bollinger-bands', 'value'),
    State('std-dev-slider', 'value'),
    State('rsi', 'value'),
    State('macd', 'value')]
)
def update_data(n_clicks, ticker, timeframe,moving_averages,show_bollinger_bands, std_dev,rsi,macd):
    
    ticker=ticker.upper()
    #df = stock.history(period=timeframe)
    time_frame_days = {
        '1m':1,
        '30m':10,
        '1d': 200,
        '1h': 20,
        '3h':20,
        '1mo': 30*60,

    }

    # Calculate the number of days of historical data to retrieve based on the user's selection
    period = f'{time_frame_days[timeframe]}d'
    if timeframe in ['1m','30m','1h','1mo']:
        interval = timeframe
    elif timeframe in ['3h']:
        interval='1h'
    else:
        interval = '1d'
    

    # Download stock data from Yahoo Finance
    stock_data = yf.download(ticker, period=period,interval=interval)
    stock_visual=yf.download(ticker,period=period,interval=interval)
    if timeframe in ['1m']:
        stock_data=stock_data.tail(250)
    if timeframe in ['1h','3h']:
        stock_data=stock_data.tail(300)


        

    
    # Update stock chart
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        row_heights=[0.7, 0.1, 0.2,0.2])
    fig.add_trace(go.Candlestick(#x=stock_data.index,
                                     open=stock_data['Open'],
                                     high=stock_data['High'],
                                     low=stock_data['Low'],
                                     close=stock_data['Close'],
                                     name='Candlestick'), row=1, col=1)
     # Calculate moving averages
    color = ['black', 'green', 'blue']
    #moving_averages = [10, 25, 50]
    i=0
    for ma in moving_averages:
        col_name = f"SMA{ma}"
        stock_visual[col_name] = stock_visual['Close'].rolling(ma).mean()
        stock_visual=stock_visual.tail(len(stock_data))

        fig.add_trace(
            go.Scatter(
                #x=stock_visual.index,
                y=stock_visual[col_name],
                name=f"{col_name}",
                line=dict(color=color[i], width=1)
            ),
            row=1, col=1
        )
        i=i+1
    if 'show' in show_bollinger_bands:
        moving_averages = [10, 25, 50]
        for ma in moving_averages:
            col_name = f"SMA{ma}"
            stock_visual[col_name] = stock_visual['Close'].rolling(ma).mean()
        stock_visual['STD'] = stock_visual['Close'].rolling(window=moving_averages[0]).std() * std_dev
        stock_visual['Bollinger High'] = stock_visual['SMA10'] + stock_visual['STD']
        stock_visual['Bollinger Low'] = stock_visual['SMA10'] - stock_visual['STD']

        fig.add_trace(
            go.Scatter(
                y=stock_visual['Bollinger High'],
                name='Bollinger High',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                y=stock_visual['Bollinger Low'],
                name='Bollinger Low',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=1, col=1
        )
    # Calculate RSI if selected
    if 'show' in rsi:
        delta = stock_data['Adj Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        stock_data['RSI'] = 100 - (100 / (1 + rs))
        fig.add_trace(go.Scatter( y=stock_data['RSI'], name='RSI'),row=3,col=1)

    # Calculate MACD if selected
    if 'show' in macd:
        exp1 = stock_data['Adj Close'].ewm(span=12, adjust=False).mean()
        exp2 = stock_data['Adj Close'].ewm(span=26, adjust=False).mean()
        stock_data['MACD'] = exp1 - exp2
        stock_data['signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
        stock_data['MACD_hist'] = stock_data['MACD'] - stock_data['signal']
        fig.add_trace(go.Scatter( y=stock_data['MACD'], name='MACD'),row=4,col=1)
        fig.add_trace(go.Scatter( y=stock_data['signal'], name='Signal'),row=4,col=1)
        fig.add_trace(go.Bar(y=stock_data['MACD_hist'], marker=dict(color=['red' if x < 0 else 'green' for x in stock_data['MACD_hist']]
    ) ,name='MACD Histogram'), row=4, col=1)



    # Add the volume bar chart to the bottom subplot
    
    fig.add_trace(go.Bar(
        #x=df.index,
        y=stock_data['Volume'],
        marker=dict(color=stock_data['Close'].diff().values >= 0, line=dict(color='red', width=1)),
        name='Volume'
    ), row=2, col=1)
    fig.update_layout(title=f'{ticker} Stock Price Chart ({timeframe})',
                      xaxis_rangeslider_visible=False,
                      height=1000# increase the height of the graph
                    )
        



    
    # Update stock data table
    df=fundamental(ticker)
    df=df.round(2)
    table=html.Table(
    [html.Tr([html.Th(col, style={'border': '1px solid black', 'border-width': '0px 0px 2px 1px', 'padding': '5px'}) for col in df.columns])] + 
    [html.Tr([html.Td(df.iloc[i][col], style={'border': '1px solid black', 'border-width': '0px 0px 0px 1px', 'padding': '5px'}) for col in df.columns]) for i in range(len(df))]
    )

 
    

    return fig, table

if __name__ == '__main__':
    app.run_server(debug=True)
