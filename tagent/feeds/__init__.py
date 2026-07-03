"""Market-data feed adapters.

`base.py` defines a normalized interface so the rest of the system never depends
on a specific broker. Today we ship an Alpaca adapter; Kiwoom or a NautilusTrader
adapter can be added later by implementing the same `MarketDataFeed` interface.
"""
