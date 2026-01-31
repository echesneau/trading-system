from trading_system.notifications.reporter import SignalReporter
from trading_system.strategies.classical import ClassicalStrategy
from trading_system.strategies.hybrid import HybridStrategy
from trading_system.data import load_yfinance_data

def test_signal_reporter(trained_model_artifacts):
    config = {
        "SAN.PA": {"ema_windows": [5, 10, 20, 50]},
        "AIR.PA": {"ema_windows": [5, 10, 20, 50]}
    }
    reporter = SignalReporter(strategy=ClassicalStrategy, data_loader=load_yfinance_data)
    report = reporter.generate_daily_report(["SAN.PA", "AIR.PA"], ticker_params=config,
                                            max_window_range=60)
    tot_len = 0
    for k in ['buy_signals', 'sell_signals', 'hold_signals', 'errors']:
        assert k in report
        assert isinstance(report[k], list)
        tot_len += len(report[k])
    assert tot_len == 2
    assert len(report['errors']) == 0
    assert 'total_tickers_analyzed' in report
    assert report['total_tickers_analyzed'] == 2

    reporter = SignalReporter(strategy=HybridStrategy, data_loader=load_yfinance_data)
    report = reporter.generate_daily_report(["SAN.PA", "AIR.PA"], ticker_params=config,
                                            max_window_range=60,
                                            model_artifacts=trained_model_artifacts, ema_windows=[5, 10, 20, 50])
    tot_len = 0
    for k in ['buy_signals', 'sell_signals', 'hold_signals', 'errors']:
        assert k in report
        assert isinstance(report[k], list)
        tot_len += len(report[k])
    assert tot_len == 2
    assert len(report['errors']) == 0
    assert 'total_tickers_analyzed' in report
    assert report['total_tickers_analyzed'] == 2