"""
Indicators Gate -- Technical Indicator Confirmation Engine.

Evaluates 6 classic indicators on raw OHLC+Volume data:
  1. RSI (14) -- oversold/overbought exit (weight: 20)
  2. CCI (20) -- alignment with RSI signal direction (weight: 15)
  3. Momentum (10) -- accelerating trend (weight: 15)
  4. ATR Volatility Range -- healthy vs extreme (weight: 15)
  5. VWAP -- price above/below volume-weighted average (weight: 20)
  6. OBV Confirmation -- no divergence with price (weight: 15)

Returns: PASS (>=75), PARTIAL (50-74), FAIL (<50)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GateResult(Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


@dataclass
class RuleVerdict:
    name: str
    weight: float
    passed: Optional[bool]
    score: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class IndicatorsOutput:
    result: GateResult
    total_score: float
    max_score: float
    percentage: float
    rules: list[RuleVerdict]
    summary: str
    indicator_values: dict


def _calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None, []
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(diff if diff > 0 else 0)
        losses.append(abs(diff) if diff < 0 else 0)
    rsi_vals = []
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        if avg_loss == 0:
            rsi_vals.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_vals.append(100.0 - (100.0 / (1.0 + rs)))
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    return rsi_vals[-1] if rsi_vals else None, rsi_vals


def _calc_cci(highs, lows, closes, period=20):
    if len(closes) < period:
        return None
    tp = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    recent_tp = tp[-period:]
    sma = sum(recent_tp) / period
    mean_dev = sum(abs(x - sma) for x in recent_tp) / period
    if mean_dev == 0:
        return 0.0
    return (tp[-1] - sma) / (0.015 * mean_dev)


def _calc_momentum(closes, period=10):
    if len(closes) < period + 1:
        return None, []
    mom_vals = []
    for i in range(period, len(closes)):
        mom_vals.append(closes[i] - closes[i - period])
    return mom_vals[-1] if mom_vals else None, mom_vals


def _calc_atr(candles, period=14):
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        h, l = candles[i].high, candles[i].low
        pc = candles[i - 1].close
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return sum(trs[-period:]) / period


def _calc_vwap(candles):
    if not candles:
        return None
    cum_pv = 0.0
    cum_vol = 0.0
    has_volume = False
    for c in candles:
        v = c.volume if c.volume else 0
        if v > 0:
            has_volume = True
        typical = (c.high + c.low + c.close) / 3
        cum_pv += typical * v
        cum_vol += v
    if cum_vol == 0 or not has_volume:
        return None
    return cum_pv / cum_vol


def _calc_obv(closes, volumes):
    if len(closes) < 2:
        return None, []
    obv = 0
    obv_vals = [0]
    has_volume = False
    for i in range(1, len(closes)):
        v = volumes[i] if volumes[i] else 0
        if v > 0:
            has_volume = True
        if closes[i] > closes[i - 1]:
            obv += v
        elif closes[i] < closes[i - 1]:
            obv -= v
        obv_vals.append(obv)
    if not has_volume:
        return None, []
    return obv, obv_vals


def _rule1_rsi(closes):
    weight = 20
    evidence = []
    rsi_now, rsi_vals = _calc_rsi(closes)
    if rsi_now is None:
        return RuleVerdict(name="RSI (14)", weight=weight, passed=False, score=0, evidence=["Insufficient data for RSI(14)"])
    evidence.append(f"RSI(14) = {rsi_now:.2f}")
    if len(rsi_vals) < 4:
        return RuleVerdict(name="RSI (14)", weight=weight, passed=False, score=0, evidence=["Need at least 4 RSI readings"])
    rsi_3_ago = rsi_vals[-4]
    passed = False
    if rsi_3_ago < 30:
        if rsi_now > rsi_3_ago and rsi_now > 30:
            passed = True
            evidence.append(f"RSI exiting oversold: {rsi_3_ago:.2f} -> {rsi_now:.2f} (above 30)")
        else:
            evidence.append(f"RSI still oversold or not recovering: {rsi_3_ago:.2f} -> {rsi_now:.2f}")
    elif rsi_3_ago > 70:
        if rsi_now < rsi_3_ago and rsi_now < 70:
            passed = True
            evidence.append(f"RSI exiting overbought: {rsi_3_ago:.2f} -> {rsi_now:.2f} (below 70)")
        else:
            evidence.append(f"RSI still overbought or not declining: {rsi_3_ago:.2f} -> {rsi_now:.2f}")
    else:
        evidence.append(f"RSI in neutral zone ({rsi_now:.2f}) -- no exit signal")
    return RuleVerdict(name="RSI (14)", weight=weight, passed=passed, score=weight if passed else 0, evidence=evidence)


def _rule2_cci(highs, lows, closes, rsi_now):
    weight = 15
    evidence = []
    cci = _calc_cci(highs, lows, closes)
    if cci is None:
        return RuleVerdict(name="CCI (20)", weight=weight, passed=False, score=0, evidence=["Insufficient data for CCI(20)"])
    evidence.append(f"CCI(20) = {cci:.2f}, RSI(14) = {rsi_now:.2f}")
    if rsi_now is not None and rsi_now < 30:
        evidence.append("RSI in oversold zone -- bearish signal")
        rsi_signal = "bearish"
    elif rsi_now is not None and rsi_now > 70:
        evidence.append("RSI in overbought zone -- bullish signal")
        rsi_signal = "bullish"
    else:
        rsi_signal = "neutral"
    if rsi_signal == "bearish":
        cci_signal = "bearish" if cci < -100 else "not bearish"
        passed = cci < -100
    elif rsi_signal == "bullish":
        cci_signal = "bullish" if cci > 100 else "not bullish"
        passed = cci > 100
    else:
        passed = True
        cci_signal = "neutral"
    if passed:
        evidence.append(f"CCI ({cci_signal}) confirms RSI ({rsi_signal}) direction -- signals aligned")
    else:
        evidence.append(f"CCI ({cci_signal}) does not confirm RSI ({rsi_signal}) direction")
    return RuleVerdict(name="CCI (20)", weight=weight, passed=passed, score=weight if passed else 0, evidence=evidence)


def _rule3_momentum(closes):
    weight = 15
    evidence = []
    mom_now, mom_vals = _calc_momentum(closes)
    if mom_now is None or len(mom_vals) < 4:
        return RuleVerdict(name="Momentum (10)", weight=weight, passed=False, score=0, evidence=["Insufficient data for Momentum(10)"])
    evidence.append(f"Momentum(10) = {mom_now:.4f}")
    m3, m2, m1 = mom_vals[-3], mom_vals[-2], mom_vals[-1]
    passed = False
    if mom_now > 0:
        if m1 > m2 > m3:
            passed = True
            evidence.append(f"Bullish momentum accelerating: {m3:.4f} -> {m2:.4f} -> {m1:.4f}")
        else:
            evidence.append(f"Bullish momentum not accelerating: {m3:.4f} -> {m2:.4f} -> {m1:.4f}")
    elif mom_now < 0:
        if m1 < m2 < m3:
            passed = True
            evidence.append(f"Bearish momentum accelerating: {m3:.4f} -> {m2:.4f} -> {m1:.4f}")
        else:
            evidence.append(f"Bearish momentum not accelerating: {m3:.4f} -> {m2:.4f} -> {m1:.4f}")
    else:
        evidence.append("Momentum is zero -- flat")
    return RuleVerdict(name="Momentum (10)", weight=weight, passed=passed, score=weight if passed else 0, evidence=evidence)


def _rule4_atr_range(candles):
    weight = 15
    evidence = []
    atr_now = _calc_atr(candles)
    if atr_now is None:
        return RuleVerdict(name="ATR Volatility Range", weight=weight, passed=False, score=0, evidence=["Insufficient data for ATR(14)"])
    all_trs = []
    for i in range(1, len(candles)):
        h, l = candles[i].high, candles[i].low
        pc = candles[i - 1].close
        tr = max(h - l, abs(h - pc), abs(l - pc))
        all_trs.append(tr)
    avg_atr = sum(all_trs[-50:]) / min(len(all_trs), 50) if all_trs else atr_now
    evidence.append(f"ATR(14) = {atr_now:.4f}, Avg ATR(50) = {avg_atr:.4f}, Ratio = {atr_now/avg_atr:.2f}x")
    passed = atr_now <= 2.0 * avg_atr
    if passed:
        evidence.append("Volatility within healthy range (<= 2x average)")
    else:
        evidence.append("Volatility is extreme (> 2x average) -- unstable conditions")
    return RuleVerdict(name="ATR Volatility Range", weight=weight, passed=passed, score=weight if passed else 0, evidence=evidence)


def _rule5_vwap(candles, closes, mom_passed, mom_now):
    weight = 20
    evidence = []
    vwap = _calc_vwap(candles)
    if vwap is None:
        return RuleVerdict(name="VWAP", weight=weight, passed=False, score=0, evidence=["Volume data missing -- cannot compute VWAP"])
    current_close = closes[-1]
    evidence.append(f"VWAP = {vwap:.4f}, Close = {current_close:.4f}")
    if not mom_passed:
        evidence.append("Momentum rule did not pass -- VWAP alignment not applicable")
        return RuleVerdict(name="VWAP", weight=weight, passed=False, score=0, evidence=evidence)
    passed = False
    if mom_now is not None and mom_now > 0:
        if current_close > vwap:
            passed = True
            evidence.append(f"Close above VWAP -- confirms bullish momentum")
        else:
            evidence.append(f"Close below VWAP -- contradicts bullish momentum")
    elif mom_now is not None and mom_now < 0:
        if current_close < vwap:
            passed = True
            evidence.append(f"Close below VWAP -- confirms bearish momentum")
        else:
            evidence.append(f"Close above VWAP -- contradicts bearish momentum")
    return RuleVerdict(name="VWAP", weight=weight, passed=passed, score=weight if passed else 0, evidence=evidence)


def _rule6_obv(closes, volumes):
    weight = 15
    evidence = []
    obv_now, obv_vals = _calc_obv(closes, volumes)
    if obv_now is None or len(obv_vals) < 6:
        return RuleVerdict(name="OBV Confirmation", weight=weight, passed=False, score=0, evidence=["Volume data missing for OBV"])
    evidence.append(f"OBV = {obv_now:.0f}")
    obv_last5 = obv_vals[-5:]
    price_last5 = closes[-5:]
    obv_up = obv_last5[-1] > obv_last5[0]
    price_up = price_last5[-1] > price_last5[0]
    passed = (obv_up and price_up) or (not obv_up and not price_up)
    if passed:
        evidence.append(f"OBV trend confirms price trend -- no divergence")
    else:
        evidence.append(f"OBV divergence: OBV vs price direction mismatch")
    return RuleVerdict(name="OBV Confirmation", weight=weight, passed=passed, score=weight if passed else 0, evidence=evidence)


def evaluate_indicators_gate(candles):
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [c.volume for c in candles]

    rsi_now, rsi_vals = _calc_rsi(closes)
    cci = _calc_cci(highs, lows, closes)
    mom_now, mom_vals = _calc_momentum(closes)
    atr_now = _calc_atr(candles)
    vwap = _calc_vwap(candles)
    obv_now, obv_vals = _calc_obv(closes, volumes)

    r1 = _rule1_rsi(closes)
    r2 = _rule2_cci(highs, lows, closes, rsi_now)
    r3 = _rule3_momentum(closes)
    r4 = _rule4_atr_range(candles)
    r5 = _rule5_vwap(candles, closes, r3.passed, mom_now)
    r6 = _rule6_obv(closes, volumes)

    verdicts = [r1, r2, r3, r4, r5, r6]
    total_score = sum(v.score for v in verdicts)
    max_score = sum(v.weight for v in verdicts)
    percentage = (total_score / max_score * 100) if max_score > 0 else 0.0

    if percentage >= 75:
        result = GateResult.PASS
        summary = f"Indicators confirm a strong directional move ({percentage:.0f}%)."
    elif percentage >= 50:
        result = GateResult.PARTIAL
        summary = f"Indicators show mixed signals ({percentage:.0f}%). Proceed with caution."
    else:
        result = GateResult.FAIL
        summary = f"Indicators do not confirm a tradeable setup ({percentage:.0f}%)."

    indicator_values = {
        "rsi": round(rsi_now, 2) if rsi_now is not None else None,
        "cci": round(cci, 2) if cci is not None else None,
        "momentum": round(mom_now, 4) if mom_now is not None else None,
        "atr_14": round(atr_now, 4) if atr_now is not None else None,
        "vwap": round(vwap, 4) if vwap is not None else None,
        "obv": round(obv_now, 0) if obv_now is not None else None,
    }

    return IndicatorsOutput(result=result, total_score=total_score, max_score=max_score,
                            percentage=round(percentage, 1), rules=verdicts, summary=summary, indicator_values=indicator_values)
