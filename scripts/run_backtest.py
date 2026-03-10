#!/usr/bin/env python
"""Main script to run backtests.

Usage:
    python -m scripts.run_backtest --config configs/backtest_config.yaml
    python scripts/run_backtest.py --config configs/backtest_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so imports work when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.engine import BacktestEngine
from src.config import BacktestConfig, load_config
from src.data.data_handler import DataHandler, HistoricalCSVDataHandler
from src.data.yfinance_handler import YFinanceDataHandler
from src.execution.execution_handler import ExecutionHandler
from src.features import (
    ATRFeature,
    BollingerBandFeature,
    FeaturePipeline,
    HigherMomentFeature,
    HurstExponentFeature,
    MACDFeature,
    ReturnFeature,
    RSIFeature,
    SMAFeature,
    VolatilityFeature,
    ZScoreFeature,
)
from src.ml import LightGBMSignalModel, MLStrategy, XGBoostSignalModel
from src.optimization.base_optimizer import PortfolioOptimizer
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.optimization.risk_parity import RiskParityOptimizer
from src.performance.metrics import PerformanceTracker
from src.performance.visualization import create_full_report
from src.portfolio.portfolio import Portfolio
from src.risk.position_sizer import (
    FixedFractionSizer,
    KellyCriterionSizer,
    PositionSizer,
    VolatilityBasedSizer,
)
from src.risk.risk_manager import RiskLimits, RiskManager
from src.strategy.base_strategy import Strategy
from src.strategy.multi_asset_sma import MultiAssetSMAStrategy
from src.strategy.sma_crossover import SMACrossoverStrategy


def build_data_handler(config: BacktestConfig) -> DataHandler:
    """Build data handler based on config.data.data_source."""
    if config.data.data_source == "yfinance":
        return YFinanceDataHandler(
            symbols=config.data.symbols,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
            cache_dir=config.data.data_path or "data/cache",
        )
    elif config.data.data_source == "csv":
        return HistoricalCSVDataHandler(
            data_path=config.data.data_path or "data/sample",
            symbols=config.data.symbols,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
        )
    else:
        msg = f"Unknown data_source: {config.data.data_source}"
        raise ValueError(msg)


def build_position_sizer(config: BacktestConfig) -> PositionSizer:
    """Build position sizer based on config.sizing."""
    method = config.sizing.method
    params = config.sizing.parameters

    if method == "fixed_fraction":
        return FixedFractionSizer(**params)
    elif method == "volatility":
        return VolatilityBasedSizer(**params)
    elif method == "kelly":
        return KellyCriterionSizer(**params)
    else:
        msg = f"Unknown sizing method: {method}"
        raise ValueError(msg)


def build_risk_manager(config: BacktestConfig) -> RiskManager | None:
    """Build risk manager based on config.risk. Returns None if disabled."""
    if not config.risk.enabled:
        return None

    limits = RiskLimits(
        max_position_pct=config.risk.max_position_pct,
        max_portfolio_exposure_pct=config.risk.max_portfolio_exposure_pct,
        max_daily_loss_pct=config.risk.max_daily_loss_pct,
        max_drawdown_pct=config.risk.max_drawdown_pct,
        max_open_positions=config.risk.max_open_positions,
        max_orders_per_day=config.risk.max_orders_per_day,
    )
    return RiskManager(limits=limits)


def build_optimizer(config: BacktestConfig) -> PortfolioOptimizer | None:
    """Build portfolio optimizer based on config.optimization.method.

    Returns None if method is 'none'.
    """
    method = config.optimization.method
    params = config.optimization.parameters

    if method == "none":
        return None
    elif method == "mean_variance":
        return MeanVarianceOptimizer(**params)
    elif method == "risk_parity":
        return RiskParityOptimizer(**params)
    else:
        msg = f"Unknown optimization method: {method}"
        raise ValueError(msg)


_FEATURE_BUILDERS = {
    "sma": lambda cfg: SMAFeature(windows=cfg.get("windows", [5, 10, 20, 50])),
    "rsi": lambda cfg: RSIFeature(period=cfg.get("period", 14)),
    "macd": lambda cfg: MACDFeature(),
    "bollinger": lambda cfg: BollingerBandFeature(window=cfg.get("window", 20)),
    "atr": lambda cfg: ATRFeature(period=cfg.get("period", 14)),
    "returns": lambda cfg: ReturnFeature(horizons=cfg.get("horizons", [1, 5, 10, 20])),
    "zscore": lambda cfg: ZScoreFeature(window=cfg.get("window", 20)),
    "higher_moments": lambda cfg: HigherMomentFeature(window=cfg.get("window", 20)),
    "hurst": lambda cfg: HurstExponentFeature(window=cfg.get("window", 100)),
    "volatility": lambda cfg: VolatilityFeature(
        windows=cfg.get("windows", [5, 20, 60])
    ),
}


def build_feature_pipeline(config: BacktestConfig) -> FeaturePipeline:
    """Build feature pipeline from config.features."""
    generators = []

    for feat_cfg in config.features.technical:
        feat_type = feat_cfg.get("type", "")
        builder = _FEATURE_BUILDERS.get(feat_type)
        if builder:
            generators.append(builder(feat_cfg))
        else:
            print(f"  Warning: unknown technical feature type '{feat_type}'")

    for feat_cfg in config.features.statistical:
        feat_type = feat_cfg.get("type", "")
        builder = _FEATURE_BUILDERS.get(feat_type)
        if builder:
            generators.append(builder(feat_cfg))
        else:
            print(f"  Warning: unknown statistical feature type '{feat_type}'")

    if not generators:
        # Default features if none specified
        generators = [
            SMAFeature(windows=[5, 10, 20, 50]),
            RSIFeature(period=14),
            MACDFeature(),
            BollingerBandFeature(window=20),
            ATRFeature(period=14),
            ReturnFeature(horizons=[1, 5, 10, 20]),
            ZScoreFeature(window=20),
            VolatilityFeature(windows=[5, 20, 60]),
        ]

    return FeaturePipeline(generators)


def build_ml_model(config: BacktestConfig) -> XGBoostSignalModel | LightGBMSignalModel:
    """Build ML model from config.ml."""
    params = dict(config.ml.parameters)

    if config.ml.model == "xgboost":
        return XGBoostSignalModel(mode=config.ml.mode, **params)
    elif config.ml.model == "lightgbm":
        return LightGBMSignalModel(mode=config.ml.mode, **params)
    else:
        msg = f"Unknown ML model: {config.ml.model}"
        raise ValueError(msg)


def build_strategy(
    config: BacktestConfig,
    optimizer: PortfolioOptimizer | None = None,
    data_handler: DataHandler | None = None,
) -> Strategy:
    """Build strategy based on config and optimizer.

    For ml_strategy: builds pipeline, trains model on historical data,
    then creates MLStrategy.

    For sma_crossover: auto-selects MultiAssetSMAStrategy when multiple
    symbols AND an optimizer are configured.
    """
    symbols = config.data.symbols
    params = dict(config.strategy.parameters)

    if config.strategy.name == "ml_strategy":
        if data_handler is None:
            msg = "ml_strategy requires data_handler for training"
            raise ValueError(msg)

        pipeline = build_feature_pipeline(config)
        model = build_ml_model(config)

        # Train on historical data — read directly from handler's internal store
        print("  Training ML model...")
        target_cfg = config.features.target
        horizon = target_cfg.get("horizon", 5)
        target_type = target_cfg.get("type", "direction")

        # Access the full historical data for training
        # (data_handler.data has all bars, not limited by current_index)
        first_symbol = symbols[0]
        if hasattr(data_handler, "data") and first_symbol in data_handler.data:
            train_data = data_handler.data[first_symbol].copy()
        else:
            msg = "Cannot access training data from data handler"
            raise ValueError(msg)

        pipeline_result = pipeline.run(
            train_data,
            target_horizon=horizon,
            target_type=target_type,
        )

        train_result = model.train(pipeline_result.features, pipeline_result.target)
        print(
            f"  Model trained: score={train_result.train_score:.3f}, "
            f"features={train_result.n_features}, "
            f"samples={train_result.n_train_samples}"
        )

        return MLStrategy(
            symbols=symbols,
            pipeline=pipeline,
            model=model,
            parameters=params,
        )

    if len(symbols) > 1 and optimizer is not None:
        params.setdefault(
            "rebalance_frequency", config.optimization.rebalance_frequency
        )
        return MultiAssetSMAStrategy(
            symbols=symbols,
            parameters=params,
            optimizer=optimizer,
        )

    return SMACrossoverStrategy(
        symbols=symbols,
        parameters=params,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating visual reports",
    )
    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Build components from config
    print("\nInitializing components...")

    data_handler = build_data_handler(config)
    position_sizer = build_position_sizer(config)
    risk_manager = build_risk_manager(config)
    optimizer = build_optimizer(config)
    strategy = build_strategy(config, optimizer, data_handler)

    portfolio = Portfolio(
        initial_capital=config.execution.initial_capital,
        symbols=config.data.symbols,
        commission_pct=config.execution.commission_pct,
        position_sizer=position_sizer,
    )

    execution_handler = ExecutionHandler(
        commission_pct=config.execution.commission_pct,
        slippage_pct=config.execution.slippage_pct,
    )

    performance_tracker = PerformanceTracker(
        initial_capital=config.execution.initial_capital,
    )

    # Create and run backtest engine
    engine = BacktestEngine(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler,
        performance_tracker=performance_tracker,
        risk_manager=risk_manager,
    )

    results = engine.run()

    # Print performance report
    performance_tracker.print_report(
        trades=results["trade_history"],
        portfolio_equity_history=portfolio.equity_history,
    )

    # Risk summary
    if risk_manager is not None:
        print("\nRisk Summary:")
        print(f"  Rejected Orders:  {results['rejected_orders']}")
        print(f"  Halted:           {risk_manager.is_halted}")
        if risk_manager.is_halted:
            print(f"  Halt Reason:      {risk_manager.halt_reason}")

    # Portfolio summary
    summary = portfolio.get_portfolio_summary()
    print("\nPortfolio Summary:")
    print(f"  Equity:           ${summary['equity']:,.2f}")
    print(f"  Cash:             ${summary['cash']:,.2f}")
    print(f"  Long Exposure:    ${summary['long_exposure']:,.2f}")
    print(f"  Short Exposure:   ${summary['short_exposure']:,.2f}")
    print(f"  Net Exposure:     ${summary['net_exposure']:,.2f}")
    print(f"  Positions:        {summary['num_positions']}")

    # Execution statistics
    exec_stats = execution_handler.get_statistics()
    print("\nExecution Statistics:")
    print(f"  Orders Processed: {exec_stats['orders_processed']}")
    print(f"  Total Commission: ${exec_stats['total_commission']:.2f}")
    print(f"  Total Slippage:   ${exec_stats['total_slippage_cost']:.2f}")

    # Config info
    print(f"\nStrategy:           {type(strategy).__name__}")
    print(f"Position Sizing:    {config.sizing.method}")
    print(f"Risk Manager:       {'enabled' if config.risk.enabled else 'disabled'}")
    opt_method = config.optimization.method
    if opt_method != "none":
        print(f"Optimization:       {opt_method}")
        print(f"Rebalance Freq:     {config.optimization.rebalance_frequency} bars")
    else:
        print("Optimization:       none")

    # Generate visual reports
    if not args.no_plots:
        print("\nGenerating visual reports...")
        equity_curve = performance_tracker.get_equity_curve()
        create_full_report(equity_curve, output_dir=args.output, show=False)


if __name__ == "__main__":
    main()
