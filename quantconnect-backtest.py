# region imports
from AlgorithmImports import *
# endregion
import numpy as np
from collections import deque

# Base URL raw GitHub untuk CSV custom data (akhiri dengan slash)
DATA_BASE_URL = "https://raw.githubusercontent.com/imsaidm/csvdata/main/"

# Helper untuk load CSV sekali di Initialize (fallback ke file lokal jika URL gagal)
def load_csv_into_dict(algo, name):
    txt = None
    url = DATA_BASE_URL + name if DATA_BASE_URL else None
    if url:
        try:
            txt = algo.Download(url)
        except Exception as e:
            algo.Debug(f"Download fail {name}: {e}")
    if not txt:
        with open(name) as f:
            txt = f.read()
    data = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("datetime"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        dt_str = parts[0].replace("Z", "")
        try:
            ts = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")
        except Exception:
            try:
                ts = datetime.strptime(dt_str, "%Y-%m-%d")
            except Exception:
                continue
        try:
            val = float(parts[1])
        except Exception:
            continue
        data[ts.date()] = val
    return data

def _daily():
    # Aman di semua env: Daily -> DAILY
    return getattr(Resolution, "Daily", getattr(Resolution, "DAILY", None))
DAILY = _daily()

class MVRVZScoreBacktest(QCAlgorithm):

    def Initialize(self):
        # Mulai 2017-09; kita pakai BTCUSD di Coinbase agar histori tersedia stabil
        self.SetStartDate(2017, 9, 1)
        self.SetEndDate(2025, 12, 31)
        self.SetAccountCurrency("USD")
        self.starting_cash = 100000
        self.SetCash("USD", self.starting_cash)
        self.SetTimeZone(TimeZones.Utc)
        self.SetBrokerageModel(BrokerageName.GDAX, AccountType.Cash)
        # Pastikan semua aset memakai harga RAW (tidak di-adjust)
        self.SetSecurityInitializer(lambda sec: sec.SetDataNormalizationMode(DataNormalizationMode.Raw))

        # BTC price from Coinbase (Coinbase/GDAX)
        self.btc = self.AddCrypto("BTCUSD", DAILY, Market.GDAX).Symbol
        self.SetBenchmark(self.btc)

        # Custom CSV data (file berada di root project; nama file = simbol + .csv)
        self.realized = self.AddData(RealizedCapData, "realizedcap", DAILY).Symbol
        self.market   = self.AddData(MarketCapData,   "marketcap",  DAILY).Symbol
        self.mvrv     = self.AddData(MVRVRatioData,   "mvrvratio",  DAILY).Symbol  # opsional, untuk logging

        # Rolling window untuk std dev
        self.lookback = 200  # pakai 200 hari supaya sinyal mulai muncul lebih cepat di backtest
        self.window = deque(maxlen=self.lookback)
        self.exit_z = 7.0        # zona merah -> keluar
        self.reenter_z = 6.0     # masuk/hold saat sudah keluar dari merah
        self.position_state = "flat"

        # Warmup biar window langsung keisi
        self.SetWarmUp(timedelta(days=self.lookback + 5))
        self.Settings.FreePortfolioValuePercentage = 0.0  # full allocation ok

        self.Debug("INIT OK")
        self.logged_ready = False
        self.trace_count = 0
        self.miss_count = 0
        self.rc_val = None
        self.mc_val = None

        # Preload CSV ke memori (dict by date) supaya tidak tergantung slice alignment
        self.rc_dict = load_csv_into_dict(self, "realizedcap.csv")
        self.mc_dict = load_csv_into_dict(self, "marketcap.csv")
        self.mr_dict = load_csv_into_dict(self, "mvrvratio.csv")

    def OnData(self, data: Slice):
        # Reset nilai harian agar tidak pakai data kadaluarsa ketika slice kosong
        self.rc_val = None
        self.mc_val = None
        today = self.Time.date()

        # Ambil nilai dari dict (lebih robust) jika ada
        if today in self.rc_dict:
            self.rc_val = self.rc_dict[today]
        else:
            if self.realized in data:
                self.rc_val = data[self.realized].Value

        if today in self.mc_dict:
            self.mc_val = self.mc_dict[today]
        else:
            if self.market in data:
                self.mc_val = data[self.market].Value

        # Ambil harga BTC (boleh dari slice atau last price)
        price_ready = self.Securities[self.btc].Price > 0

        if self.rc_val is None or self.mc_val is None or not price_ready:
            if self.miss_count < 3 and not self.IsWarmingUp:
                self.Debug(f"MISSING {self.Time.date()} rc:{self.rc_val is not None} mc:{self.mc_val is not None} px:{price_ready}")
                self.miss_count += 1
            return

        realized_cap = self.rc_val
        market_cap   = self.mc_val
        if market_cap <= 0 or realized_cap <= 0:
            return

        diff = market_cap - realized_cap
        self.window.append(diff)

        # Tunggu window cukup agar std dev stabil (boleh ganti 60/90 kalau datamu pendek)
        if len(self.window) < self.lookback:
            return

        std_dev = np.std(self.window)
        if std_dev == 0:
            return

        z = diff / std_dev

        # Debug ringan tiap awal bulan
        if self.Time.day == 1:
            self.Debug(f"{self.Time.date()} | Z={z:.2f} | MC={market_cap:.0f} | RC={realized_cap:.0f}")
        if not self.logged_ready and not self.IsWarmingUp:
            self.Debug(f"READY {self.Time.date()} Z={z:.2f}")
            self.logged_ready = True
        if self.logged_ready and self.trace_count < 5:
            self.Debug(f"TRACE {self.Time.date()} Z={z:.2f} MC={market_cap:.0f} RC={realized_cap:.0f}")
            self.trace_count += 1

        # Plot biar gampang lihat di chart QC
        self.Plot("MVRV", "Z-Score", z)
        if today in self.mr_dict:
            self.Plot("MVRV", "MVRV Ratio", self.mr_dict[today])
        elif self.mvrv in data:
            self.Plot("MVRV", "MVRV Ratio", data[self.mvrv].Value)

        # Aturan: zona merah => keluar; selain itu => hold BTC
        invested = self.Portfolio[self.btc].Invested

        if z >= self.exit_z and invested:
            self.Liquidate(self.btc, tag=f"Exit RED Z {z:.2f}")
            self.position_state = "flat"
        elif z <= self.reenter_z and not invested and not self.IsWarmingUp:
            # Pakai 95% supaya ada buffer fee/slippage -> hindari insufficient buying power
            self.SetHoldings(self.btc, 0.95, tag=f"Enter Z {z:.2f}")
            self.position_state = "long"


# ============================
# CUSTOM DATA CLASSES
# ============================

class RealizedCapData(PythonData):
    def GetSource(self, config, date, isLive):
        # coba remote GitHub raw; fallback ke file lokal bila URL kosong
        if DATA_BASE_URL:
            return SubscriptionDataSource(
                DATA_BASE_URL + "realizedcap.csv",
                SubscriptionTransportMedium.RemoteFile,
                FileFormat.CSV)
        filename = f"{config.Symbol.Value}.csv"  # expect realizedcap.csv
        return SubscriptionDataSource(filename,
                                      SubscriptionTransportMedium.LocalFile,
                                      FileFormat.CSV)
    def Reader(self, config, line, date, isLive):
        if not line.strip(): return None
        parts = line.split(',')
        if len(parts) < 2 or parts[0] == "Datetime": return None
        d = RealizedCapData()
        d.Symbol = config.Symbol
        d.Time   = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%SZ")
        d.Value  = float(parts[1])
        d.EndTime = d.Time + timedelta(days=1)
        return d

class MarketCapData(PythonData):
    def GetSource(self, config, date, isLive):
        if DATA_BASE_URL:
            return SubscriptionDataSource(
                DATA_BASE_URL + "marketcap.csv",
                SubscriptionTransportMedium.RemoteFile,
                FileFormat.CSV)
        filename = f"{config.Symbol.Value}.csv"  # expect marketcap.csv
        return SubscriptionDataSource(filename,
                                      SubscriptionTransportMedium.LocalFile,
                                      FileFormat.CSV)
    def Reader(self, config, line, date, isLive):
        if not line.strip(): return None
        parts = line.split(',')
        if len(parts) < 2 or parts[0] == "Datetime": return None
        d = MarketCapData()
        d.Symbol = config.Symbol
        d.Time   = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%SZ")
        d.Value  = float(parts[1])
        d.EndTime = d.Time + timedelta(days=1)
        return d

class MVRVRatioData(PythonData):
    def GetSource(self, config, date, isLive):
        if DATA_BASE_URL:
            return SubscriptionDataSource(
                DATA_BASE_URL + "mvrvratio.csv",
                SubscriptionTransportMedium.RemoteFile,
                FileFormat.CSV)
        filename = f"{config.Symbol.Value}.csv"  # expect mvrvratio.csv
        return SubscriptionDataSource(filename,
                                      SubscriptionTransportMedium.LocalFile,
                                      FileFormat.CSV)
    def Reader(self, config, line, date, isLive):
        if not line.strip(): return None
        parts = line.split(',')
        if len(parts) < 2 or parts[0] == "Datetime": return None
        d = MVRVRatioData()
        d.Symbol = config.Symbol
        d.Time   = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%SZ")
        d.Value  = float(parts[1])
        d.EndTime = d.Time + timedelta(days=1)
        return d