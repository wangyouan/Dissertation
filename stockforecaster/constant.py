#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: Dissertation
# File name: constant
# Author: Mark Wang
# Date: 30/5/2016


class Constants(object):
    PRICE_TYPE = "price type"
    DATE = "Date"
    DIRECTION_PREDICTION = 'Direction'
    AMOUNT_PREDICTION = 'Amount'
    TARGET_PRICE = 'Target'
    TODAY_PRICE = 'TodayPrice'
    CHANGE_DIRECTION = 'ChangeDirection'
    CHANGE_AMOUNT = 'ChangeAmount'
    SPARK = 'Spark'
    TENSORFLOW = 'TensorFlow'

    # information used in stock price info
    STOCK_PRICE = "stock_price"
    STOCK_HIGH = "High"
    STOCK_LOW = "Low"
    STOCK_OPEN = "Open"
    STOCK_CLOSE = "Close"
    STOCK_VOLUME = "Volume"
    STOCK_ADJUSTED_CLOSED = "Adj Close"
    DATA_PERIOD = "data_period"

    # information for indicators info
    TECHNICAL_INDICATOR = "technical_indicator"

    # For macd info
    MACD = "MACD"
    MACD_SLOW_PERIOD = "MACD slow period"
    MACD_FAST_PERIOD = "MACD fast period"
    MACD_TIME_PERIOD = "MACD time period"

    # For moving average info
    SMA = "SMA"  # "simple moving average"
    EMA = "EMA"  # "Exponential moving average"
    ROC = "ROC"  # "Rate of change"
    RSI = "RSI"  # "Relative strength index"

    # Fundamental Analysis
    FUNDAMENTAL_ANALYSIS = "fundamental analysis"

    # Bond, index and ETF
    BOND = "Bond"
    US10Y_BOND = "CBOE Interest Rate 10 Year T No"  # Symbol ^TNX
    US30Y_BOND = "Treasury Yield 30 Years"  # Symbol ^TYX
    HSI = "Hang Seng Index"  # ^HSI
    FXI = "iShares China Large-Cap"
    IC = "2801.HK"  # iShares China
    IA = "iShares CSI A-Share Financials"  # 2829.HK
    IA300 = "iShares CSI 300 A-Share"  # 2846.HK
    IMSCI = "iShares MSCI AC Asia ex Japan"  # 3010.HK
    SHSE = "000001.SS"  # Shanghai stock exchange market

    # Normalization method
    MIN_MAX = "Min-Max Normalization"
    SIGMOID = "Sigmoid Normalization"

    # define constants used in get loss function
    LSE = 'least square error'
    LAE = 'least absolute error'

    # define method type used in neural network regression type
    BP_SGD = 'back propagation with stochastic gradient descent'
    BP = 'back propagation'

    # define method used in linear regression
    GD = 'Standard gradient descent'

    # Fundamental analysis type
    FA_RAW_DATA = "raw data"
    FA_RATIO = "ratio"
    FA_NORMALIZATION = "Normalization"

    # Used in currency exchange
    CURRENCY_EXCHANGE = 'CURRENCY_EXCHANGE'
    HKD = "HKD"  # "Hong Kong Dollar"
    AUD = "AUD"  # "Australian Dollar"
    USD = "USD"  # "United States Dollar"
    CNY = "CNY"  # "China Yuan"
    EUR = "EUR"  # "European Euro"

    FROM = "from"
    TO = "to"

    GOLDEN_PRICE = "gold price in CNY"
    TYPE_GOLDEN_PRICE = 'GoldenPrice'

    ARTIFICIAL_NEURAL_NETWORK = "Artificial_Neural_Network"
    RANDOM_FOREST = "Random_Forest"
    LINEAR_REGRESSION = "Linear_Regression"
    NAIVE_BAYES = 'Naive_Bayes'
    LOGISTIC_REGRESSION = 'Logistic_Regression'
    SVM = 'SVM'

    PCA = "Principal Component Analysis"
    STANDARD_SCALE = "Standard Scaler"

    # The following are used in in HKAB HKD Interest Settlement Rates
    HIBOR = 'HIBOR'
    OVER_NIGHT = "Overnight"
    ONE_WEEK = "1 Week"
    ONE_MONTH = "1 Month"
    TWO_MONTHS = "2 Months"
    THREE_MONTHS = "3 Months"
    HALF_YEAR = "6 Months"
    ONE_YEAR = "12 Months"

    UNITED_STATES = "United States"
    HONG_KONG = "Hong Kong"

    # parameters used in data saving
    SAVE_TYPE_MODEL = "model"
    SAVE_TYPE_INPUT = 'input'
    SAVE_TYPE_OUTPUT = 'output'

    RATIO_AMOUNT = 'change ratio'
    RAW_AMOUNT = 'raw amount'
    ORIGINAL_PRICE = 'original price'
