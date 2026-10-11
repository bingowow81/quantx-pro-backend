import os
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import anthropic
from cachetools import TTLCache

app = FastAPI(title="QUANTX PRO Institutional Core Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
quant_cache = TTLCache(maxsize=300, ttl=600)  # 10분 캐시 (중복 연산 및 토큰 방어)

SYSTEM_PROMPT = """
당신은 기관 퀀트 헤지펀드의 수석 포트폴리오 매니저이자 고지능 대화형 어시스턴트 'QUANTX PRO'입니다.

[운영 가이드라인]
1. [일상적 대화 및 일반 질의]
- 인삿말, 일반 상식, 투자 철학 등 일상적인 질문에는 친절하고 명민한 대화형 파트너로서 유연하게 대화하십시오.
- 이때는 억지로 4단계 퀀트 리포트 서식을 쓰지 말고 자연스러운 줄글로 응답하십시오.

2. [금융 / 종목 / 투자 전략 질의]
- 데이터가 제공되었거나 주식/시장 분석 질의가 들어오면 철저히 정량 데이터에 기반해 냉철하게 진단하십시오.
- 감정적 수식어와 이모티콘은 배제하고, 반드시 다음 4단계 양식으로 출력하십시오:

[1단계: 종합 퀀트 스코어 및 포지션 의견]
- 종합 매력도 점수 (100점 만점 수치)
- 권고 포지션 (적극 매수 / 분할 매수 / 중립 관망 / 비중 축소)
- 핵심 요약 (2문장 이내)

[2단계: 다중 팩터 정밀 진단]
- 추세 및 이평선: 5일, 20일, 60일 이평선 정배열/역배열 및 이격도
- 모멘텀: RSI 수치 기반 과매수/과매도 및 반등 탄력
- 2024 백테스트 통계 및 뉴스 모멘텀 평가

[3단계: 정량적 매매 시나리오 가이드]
- 1차 매수가 및 지지선 기반 손절가 (구체적 가격 명시)
- 1차 목표가 및 리스크 대비 수익 비율 (Risk/Reward Ratio 1:1.5 이상)

[4단계: 핵심 하방 리스크 요인 2가지]

말미에는 반드시 아래 문구를 추가하십시오:
"※ 본 분석은 정량 알고리즘 시뮬레이션 기반 참고 자료이며 원금 손실 위험이 있습니다."
"""

class StockRequest(BaseModel):
    symbol: str

class ChatRequest(BaseModel):
    message: str
    symbol_context: Optional[str] = ""

def run_2024_backtest(df_full: pd.DataFrame) -> Dict[str, Any]:
    """2024년 1년치 5MA-20MA 골든크로스 전략 백테스트 순수 파이썬 연산"""
    try:
        df_2024 = df_full.loc["2024-01-01":"2024-12-31"].copy()
        if len(df_2024) < 30:
            return {"status": "데이터 부족", "total_return": 0.0, "mdd": 0.0, "win_rate": 0.0}

        df_2024["MA5"] = df_2024["Close"].rolling(5).mean()
        df_2024["MA20"] = df_2024["Close"].rolling(20).mean()

        # 신호 생성: 5일선 > 20일선 보유(1), 이하 매도(0)
        df_2024["Signal"] = np.where(df_2024["MA5"] > df_2024["MA20"], 1, 0)
        df_2024["Daily_Return"] = df_2024["Close"].pct_change().fillna(0)
        # 익일 시그널 적용
        df_2024["Strat_Return"] = df_2024["Signal"].shift(1).fillna(0) * df_2024["Daily_Return"]

        # 누적 수익률 계산
        cum_ret = (1 + df_2024["Strat_Return"]).cumprod()
        total_return = float((cum_ret.iloc[-1] - 1) * 100)

        # MDD(최대 낙폭)
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        mdd = float(drawdown.min() * 100)

        # 승률
        trades = df_2024[df_2024["Strat_Return"] != 0]["Strat_Return"]
        win_rate = float((len(trades[trades > 0]) / len(trades) * 100)) if len(trades) > 0 else 0.0

        return {
            "total_return": round(total_return, 2),
            "mdd": round(abs(mdd), 2),
            "win_rate": round(win_rate, 2),
            "trade_days": int(len(df_2024))
        }
    except Exception:
        return {"total_return": 0.0, "mdd": 0.0, "win_rate": 0.0, "trade_days": 0}

def fetch_comprehensive_stock_data(query: str) -> Dict[str, Any]:
    ticker_sym = query.strip()
    if ticker_sym.isdigit() and len(ticker_sym) == 6:
        ticker_sym = f"{ticker_sym}.KS"

    ticker = yf.Ticker(ticker_sym)
    hist = ticker.history(period="2y")

    if hist.empty and ticker_sym.endswith(".KS"):
        ticker_sym = ticker_sym.replace(".KS", ".KQ")
        ticker = yf.Ticker(ticker_sym)
        hist = ticker.history(period="2y")

    if hist.empty or len(hist) < 30:
        raise ValueError(f"'{query}' 종목의 시장 데이터를 수집할 수 없습니다.")

    # 1. 가격 및 기본 변동률
    current_price = float(hist["Close"].iloc[-1])
    prev_close = float(hist["Close"].iloc[-2])
    daily_change_pct = ((current_price - prev_close) / prev_close) * 100

    hist_30d = hist.tail(30)
    return_20d = ((current_price - float(hist["Close"].iloc[-21])) / float(hist["Close"].iloc[-21])) * 100

    # 2. 이동평균선
    ma5 = float(hist["Close"].rolling(5).mean().iloc[-1])
    ma20 = float(hist["Close"].rolling(20).mean().iloc[-1])
    ma60 = float(hist["Close"].rolling(60).mean().iloc[-1])

    # 3. RSI(14)
    delta = hist["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14, min_periods=1).mean().iloc[-1]
    avg_loss = loss.rolling(14, min_periods=1).mean().iloc[-1]
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = float(100 - (100 / (1 + rs)))

    # 4. 2024년 백테스트 실행
    backtest_2024 = run_2024_backtest(hist)

    # 5. 뉴스 헤드라인 수집 (상위 3건)
    news_items = []
    try:
        raw_news = ticker.news
        if raw_news:
            for n in raw_news[:3]:
                title = n.get("title") or n.get("content", {}).get("title", "")
                if title:
                    news_items.append(title)
    except Exception:
        news_items = ["최신 뉴스 피드 연결 대기"]

    # 6. 차트용 30일 시계열 데이터
    chart_dates = [d.strftime("%m-%d") for d in hist_30d.index]
    chart_prices = [round(float(p), 2) for p in hist_30d["Close"].tolist()]

    # 7. 퀀트 종합 스코어 알고리즘
    tech_score = int(max(15, min(95, 100 - abs(rsi - 50) * 1.4)))
    trend_bonus = 10 if ma5 > ma20 > ma60 else (-10 if ma5 < ma20 < ma60 else 0)
    score = int(max(10, min(98, (tech_score * 0.5) + 35 + trend_bonus + (backtest_2024["total_return"] * 0.1))))

    currency = "원" if ".K" in ticker_sym else "USD"

    return {
        "name": query,
        "symbol": ticker_sym,
        "current_price": round(current_price, 2),
        "currency": currency,
        "daily_change_pct": round(daily_change_pct, 2),
        "return_20d": round(return_20d, 2),
        "rsi": round(rsi, 2),
        "ma5": round(ma5, 2),
        "ma20": round(ma20, 2),
        "ma60": round(ma60, 2),
        "score": score,
        "technical_score": tech_score,
        "backtest_2024": backtest_2024,
        "news": news_items,
        "chart_data": {
            "labels": chart_dates,
            "prices": chart_prices
        }
    }

@app.post("/api/analyze")
async def analyze_stock(req: StockRequest):
    query = req.symbol.strip()
    if not query:
        raise HTTPException(status_code=400, detail="종목명을 입력하십시오.")

    if query in quant_cache:
        return quant_cache[query]

    try:
        data = fetch_comprehensive_stock_data(query)
        quant_cache[query] = data
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    user_msg = req.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="질의 내용을 입력하십시오.")

    prompt_content = user_msg
    if req.symbol_context:
        prompt_content = f"[실시간 수집 퀀트 지표]\n{req.symbol_context}\n\n[사용자 질의]\n{user_msg}"

    def generate_stream():
        try:
            with client.messages.stream(
                model="claude-3-5-haiku-20241022",
                max_tokens=1800,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt_content}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            yield f"\n[엔진 연산 통신 에러: {str(e)}]"

    return StreamingResponse(generate_stream(), media_type="text/plain")
