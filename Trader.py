import time, datetime
import math
from binance.client import Client
import pandas as pd
import numpy as np
import sys
import configparser


class Trader(Client):
    def __init__(self, api_key, api_secret, live_trading):
        super(Trader, self).__init__(api_key, api_secret)
        self.live_trading = live_trading
        self.time_offset = self.get_time_offset()
        self.exchange_info = self.get_exchange_info()

    def _request(self, method, uri, signed, force_params=False, **kwargs):
        ### **Developer's note** This function has to be overwritten because it is causing errors when there is a time delay between 
        # the binance server and the local server

        # set default requests timeout
        kwargs['timeout'] = 10

        # add our global requests params
        if self._requests_params:
            kwargs.update(self._requests_params)

        data = kwargs.get('data', None)
        if data and isinstance(data, dict):
            kwargs['data'] = data

            # find any requests params passed and apply them
            if 'requests_params' in kwargs['data']:
                # merge requests params into kwargs
                kwargs.update(kwargs['data']['requests_params'])
                del(kwargs['data']['requests_params'])

        if signed:
            # generate signature
            kwargs['data']['timestamp'] = int(time.time() * 1000 + self.time_offset)
            kwargs['data']['signature'] = self._generate_signature(kwargs['data'])

        # sort get and post params to match signature order
        if data:
            # sort post params
            kwargs['data'] = self._order_params(kwargs['data'])
            # Remove any arguments with values of None.
            null_args = [i for i, (key, value) in enumerate(kwargs['data']) if value is None]
            for i in reversed(null_args):
                del kwargs['data'][i]

        # if get request assign data array to params value for requests lib
        if data and (method == 'get' or force_params):
            kwargs['params'] = '&'.join('%s=%s' % (data[0], data[1]) for data in kwargs['data'])
            del(kwargs['data'])

        self.response = getattr(self.session, method)(uri, **kwargs)
        return self._handle_response()

    def get_time_offset(self):
        # Getting the time difference between the binance server time and the local server
        st = int(self.get_server_time()["serverTime"])
        mt = int(time.time()*1000)
        return st - mt

    def get_precision(self, symbol, type='stepSize'):
        infos = self.exchange_info["symbols"]
        for info in infos:
            if info["baseAsset"] == symbol[:len(info["baseAsset"])]:
                data = info
                break
        if type == "stepSize":
            tick = data["filters"][2]["stepSize"]
            precision = math.floor(-math.log10(float(tick)))
            return precision
        if type == "tickSize":
            tick = data["filters"][0]["tickSize"]
            precision = math.floor(-math.log10(float(tick)))
            return precision
        elif type == "base":
            return data["baseAssetPrecision"]
        elif type == "quote":
            return data["quotePrecision"]

    def place_market_buy(self, symbol, quantity):
        data = self.order_market_buy(symbol=symbol, quantity=quantity)
        return data

    def place_market_sell(self, symbol, quantity):
        data = self.order_market_sell(symbol=symbol, quantity=quantity)
        return data

    def get_balance(self, asset):
        balance = float(self.get_asset_balance(asset)['free'])
        return balance

    def round_down_safely(self, x, precision):
        # rounds down safely by avoiding floating point errors
         return round(math.floor(x * (10**precision)) * 10**-precision, precision)

    def place_market_order(self, base_asset="BTC", quote_asset="USDT", action="sell", percentage=100):
        # Get base asset and quote asset
        symbol = base_asset + quote_asset

        # Determine transaction quantity
        precision = self.get_precision(symbol, "stepSize")
        
        if action == 'sell':
            asset = base_asset
        elif action == 'buy':
            asset = quote_asset
        balance = self.get_balance(asset)

        if action == 'sell':
            quantity = self.round_down_safely(balance * percentage / 100, precision)
        elif action == 'buy':
            last_price = float(self.get_symbol_ticker(symbol=symbol)['price'])
            print("last price", last_price)
            quantity = self.round_down_safely(balance * percentage / 100 / last_price, precision)

        #print("Balance = {}, Percentage = {}, Precision = {}, Quantity = {}".format(balance, percentage, precision, quantity))
        assert quantity > 0.0, "Balance of {} {} is too low to execute trade.".format(balance, asset)

        # Execute trade
        if self.live_trading:
            if action == 'sell':
                response = self.place_market_sell(symbol, quantity)
            elif action == 'buy':
                response = self.place_market_buy(symbol, quantity)
            else:
                raise NotImplementedError('Action {} is not implemented.'.format(action))
        else:
            response = {'status': 'Live trading is off'}
        
        # Log
        msg = "Placing {} order for {} of {}".format(action, quantity, symbol)
        print(msg)
        print(response)
        return response['status'] == 'FILLED'


    ### Related to trading strategy ###
    # 1 - Get kline data for past 1000 candles
    # 2 - Build Heikin-Ash candlesticks and colors
    # 3 - Make trade

    # Other points
    # - Start out in a state (which pair, current position)
    # - Maintain the state (if in buy or sell)
    # - Log trades
	# - If there are any errors, stop trading to prevent unexpected behavior
    
    def start_bot(self, base_asset, quote_asset, initial_position, interval='2h', wait_time=60):
        '''EXAMPLE:
        base_asset = 'BTC'
        quote_asset = 'USDT'
        initial_position = 'short' // Can be 'short' or 'long' and tells the bot where you are starting off
        interval = '1h' // Can be any of {1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M}
        wait_time = 60 // Seconds between API queries
        '''
        print("Bot Starting at {}".format(datetime.datetime.today()))
        print("Base Asset: {} Quote Asset: {} Interval: {} Wait Time: {}".format(base_asset, quote_asset, interval, wait_time))
        position = initial_position
        print("Initial Position: {}".format(position))
        symbol = base_asset + quote_asset
        while True:
            ## Get data from API
            num_rows = 1000
            candles = t.get_klines(symbol=symbol, interval=interval, limit=num_rows)
            df = pd.DataFrame(candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            num_rows = df.shape[0] # Make sure that num_rows is correct in case 1000 periods haven't happened yet

            ## Clean data
            df = df[['open_time', 'open', 'high', 'low', 'close', 'close_time']]
            df['open'] = df['open'].astype('float')
            df['close'] = df['close'].astype('float')
            df['high'] = df['high'].astype('float')
            df['low'] = df['low'].astype('float')

            ## Calculate Heikin-Ashi values (source: https://school.stockcharts.com/doku.php?id=chart_analysis:heikin_ashi)
            # HA Close is based on current open close and low
            df['HA_close'] = (df['open'] + df['close'] + df['low'] + df['close']) / 4
            # Set HA values to regular open and close for first period in data
            df.at[0, 'HA_open'] = df['open'].iloc[0]
            df.at[0, 'HA_close'] = df['close'].iloc[0]
            # Set HA_open with recursive formula
            for i in range(1, num_rows):
                df.at[i, 'HA_open'] = (df.loc[i-1, 'HA_open'] + df.loc[i-1, 'HA_close']) / 2
            # Set HA_high and HA_low
            df['HA_high'] = df[['high', 'HA_open', 'HA_close']].max(axis=1)
            df['HA_low'] = df[['low', 'HA_open', 'HA_close']].min(axis=1)
            # Set color of HA candle (True = Green) if HA_close=HA_open, take previous color
            for i in range(0, df.shape[0]):
                d = df.at[i, 'HA_close'] > df.at[i, 'HA_open']
                if i > 0 and df.at[i, 'HA_close'] == df.at[i, 'HA_open']:
                    d = df.at[i-1, 'HA_green_candle']
                df.at[i, 'HA_green_candle'] = d
            
            ## Execute trading logic
            last_candle = df['HA_green_candle'].iloc[-2]
            second_to_last_candle = df['HA_green_candle'].iloc[-3]
            current_price = df['close'].iloc[-1]

            if last_candle == second_to_last_candle:
                signal = 'none'
            elif last_candle == True:
                signal = 'buy'
            elif last_candle == False:
                signal = 'sell'

            ## Make trade
            print("{}\tSignal: {}\tPosition: {}\tCurrent Price: {}".format(datetime.datetime.today(), signal, position, current_price))
            if signal != 'none':
                if position == 'not_in_trade' and signal == 'buy':
                    self.place_market_order(base_asset=base_asset, quote_asset=quote_asset, action='buy')
                    position = 'in_trade'
                elif position == 'in_trade' and signal == 'sell':
                    self.place_market_order(base_asset=base_asset, quote_asset=quote_asset, action='sell')
                    position = 'not_in_trade'
            
            ## Wait
            time.sleep(wait_time)

if __name__ == '__main__':
    ## Get configuration data
    config_fname = sys.argv[1]
    config = configparser.ConfigParser()
    config.sections()
    config.read(config_fname)
    assert config['TRADING']['LIVE_TRADING'] in ['True', 'False'], 'Configuration value for LIVE_TRADING ({}) is invalid. Must be True or False.'.format(config['TRADING']['LIVE_TRADING'])
    assert config['TRADING']['INITIAL_POSITION'] in ['in_trade', 'not_in_trade'], 'Configuration value for INITIAL_POSITION ({}) is invalid. Must be in_trade or not_in_trade.'.format(config['TRADING']['INITIAL_POSITION'])

    ## Start trading
    t = Trader(config['API_KEYS']['API_KEY'], 
                config['API_KEYS']['API_SECRET'], 
                live_trading=config['TRADING']['LIVE_TRADING']=='True'
                )

    t.start_bot(config['TRADING']['BASE_ASSET'], 
                config['TRADING']['QUOTE_ASSET'],
                config['TRADING']['INITIAL_POSITION'],
                interval=config['TRADING']['INTERVAL'],
                wait_time=int(config['TRADING']['WAIT_TIME']),
                )
