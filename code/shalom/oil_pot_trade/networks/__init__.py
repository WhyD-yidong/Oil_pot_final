import os

if os.environ.get('RLTRADER_BACKEND', 'pytorch') == 'pytorch':
    from shalom.oil_pot_trade.networks.networks_pytorch import Network, LSTMNetwork
else:
    from shalom.oil_pot_trade.networks.networks_keras import Network, DNN, LSTMNetwork, CNN

__all__ = [
    'Network', 'DNN', 'LSTMNetwork', 'CNN'
]
