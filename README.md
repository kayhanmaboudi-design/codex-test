#نسخه نهایی بدون تغییر کد GPT
# -*- coding: utf-8 -*-
#"""
#مدل OLG سه‌دوره‌ای نظام بازنشستگی ایران با تفکیک جنسیت و قانون ساده سرمایه (K)
#===========================================================================

 #ین فایل، نسخه نهایی و کامل کد پژوهش است؛ مطابق تمام دستوراتی که تا الان مطرح شد:
 # • سه گروه سنی مطابق فایل مرکز آمار ایران: 15-29 ،30-64 ،65+
 # • تفکیک جنسیتی (M / F) برای جمعیت، مشارکت، مزد و مستمری
 # • سه سناریوی سیاستی: (1) روند پایه، (2) اصلاح سن بازنشستگی، (3) اصلاح نرخ حق بیمه،
 #   + یک سناریوی اختیاری حمایت دولت از کسری صندوق
 # • قانون ساده و بدون دردسر برای سرمایه K_t تا باگ نگیرد و نیاز به داده اضافی نباشد:
 #     K_t = θ_K * WageBill_{t-1}
 #   یعنی سهم ثابتی از مجموع دستمزدِ شاغلان سال قبل، به‌عنوان موجودی سرمایه سال جاری در نظر گرفته می‌شود.
 #   (اگر داده سال قبل موجود نبود، از همان سال استفاده و یا مقدار اولیه تعیین می‌شود.)
 # • پر کردن NaN ها و کنترل واحدها
 # • توضیحات فارسی دقیق در کل کد

#نکته: اگر بعداً خواستی قانون سرمایه یا هر جزء دیگری را تغییر دهی، فقط بخش مشخص‌شده را اصلاح کن.

#روش استفاده:
# 1) فایل‌های اکسل را طبق قالب توضیح‌شده آماده و در کنار این اسکریپت قرار بده.
# 2) مقادیر پایه (مثل نرخ بیمه، سن بازنشستگی اولیه) را در فایل‌های مربوطه بگذار.
# 3) یکی از سناریوها را اجرا کن و خروجی DataFrame را بررسی/ذخیره/نمودارسازی کن.

#مواردی که باید پر شوند:
 # population_cohort_sex.xlsx
 # participation_cohort_sex.xlsx
 # wage_cohort_sex.xlsx
 # pension_cohort_sex.xlsx
 # productivity_growth.xlsx
 # capital_return.xlsx     (اختیاری؛ در این نسخه استفاده مستقیم نمی‌شود اما نگه داشته‌ایم)
 # premium_rate.xlsx       (فقط مقدار سال پایه لازم است)
 # ret_age_m.xlsx          (فقط مقدار سال پایه لازم است)
 # ret_age_w.xlsx          (فقط مقدار سال پایه لازم است)

#(در صورت نداشتن فایل savings_cohort_sex.xlsx می‌توان آن را خالی گذاشت، چون K بر اساس قانون θ_K تعیین می‌شود.)
#"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys, pandas as pd
print("RUNNING FILE:", __file__)
print("PANDAS VERSION:", pd.__version__)


# ----------------------------
# 1. تنظیمات و پارامترها
# ----------------------------
START_YEAR = 1400
END_YEAR   = 1430
YEARS  = np.arange(START_YEAR, END_YEAR+1)
COHORTS = ['15-29', '30-64', '65+']
SEXES   = ['M', 'F']

# پارامترهای اقتصادی «قابل تغییر»
BETA   = 0.96   # نرخ ترجیحات زمانی (اگر لازم نشد، تاثیری ندارد)
ALPHA  = 0.35   # سهم سرمایه در تولید کاب-داگلاس
A_0    = 1.0    # سطح اولیه بهره‌وری کل عوامل
THETA_K = 0.15  # سهم ثابت از صورت‌حساب دستمزد شاغلانِ سال قبل که به سرمایه تبدیل می‌شود

# فولدر داده‌ها (در صورت نیاز)
DATA_DIR = Path('.')

# ----------------------------
# 2. بارگذاری داده‌ها از اکسل
# ----------------------------
# قالب فایل‌ها: سطر = سال (1400..1430) ، ستون‌ها = 15-29_M, 15-29_F, 30-64_M, 30-64_F, 65+_M, 65+_F
# همه فایل‌ها باید این ستون‌ها را داشته باشند (یا حداقل آن‌هایی که لازم است).

def load_df(name):
    """تابع کمکی برای بارگذاری یک دیتافریم از اکسل و اطمینان از ایندکس عددی سال"""
    df = pd.read_excel(DATA_DIR / f"{name}.xlsx", index_col=0)
    # تبدیل ایندکس به int (در صورت نیاز)
    df.index = df.index.astype(int)
    # مرتب‌سازی بر اساس سال
    df = df.sort_index()
    return df

# داده‌های ضروری
population     = load_df('population_cohort_sex')      # نفر
participation  = load_df('participation_cohort_sex')   # 0 تا 1
wage           = load_df('wage_cohort_sex')            # ریال یا تومان (یکسان)
pension        = load_df('pension_cohort_sex')         # ریال یا تومان (یکسان)

# داده‌های اختیاری
try:
    savings = load_df('savings_cohort_sex')            # اگر موجود بود
except FileNotFoundError:
    savings = pd.DataFrame(0, index=YEARS, columns=[f"{c}_{s}" for c in COHORTS for s in SEXES])

# سری‌های سالانه
productivity_growth = pd.read_excel(DATA_DIR / 'productivity_growth.xlsx', index_col=0).squeeze("columns")
productivity_growth.index = productivity_growth.index.astype(int)

# بازده سرمایه نگه می‌داریم (فعلاً در نرخ بازده تعادلی داخلی استفاده نمی‌کنیم)
try:
    capital_return = pd.read_excel(DATA_DIR / 'capital_return.xlsx', index_col=0).squeeze("columns")
    capital_return.index = capital_return.index.astype(int)
except FileNotFoundError:
    capital_return = pd.Series(0.08, index=YEARS)  # مقدار پیش‌فرض 8%

premium_rate = pd.read_excel(DATA_DIR / 'premium_rate.xlsx', index_col=0).squeeze("columns")
premium_rate.index = premium_rate.index.astype(int)
ret_age_m    = pd.read_excel(DATA_DIR / 'ret_age_m.xlsx', index_col=0).squeeze("columns")
ret_age_m.index = ret_age_m.index.astype(int)
ret_age_w    = pd.read_excel(DATA_DIR / 'ret_age_w.xlsx', index_col=0).squeeze("columns")
ret_age_w.index = ret_age_w.index.astype(int)

# هزینه‌های اضافی صندوق (اختیاری)
try:
    pension_other = pd.read_excel(DATA_DIR / 'pension_other.xlsx', index_col=0).squeeze("columns")
    pension_other.index = pension_other.index.astype(int)
except FileNotFoundError:
    pension_other = pd.Series(0, index=YEARS)

# ----------------------------
# 3. پاک‌سازی اولیه داده‌ها (جلوگیری از NaN)
# ----------------------------
all_cols = [f"{c}_{s}" for c in COHORTS for s in SEXES]
for df in [population, participation, wage, pension, savings]:
    # اطمینان از داشتن همه ستون‌ها
    for col in all_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[all_cols]
    df.fillna(0, inplace=True)

productivity_growth = productivity_growth.reindex(YEARS).fillna(0)
capital_return      = capital_return.reindex(YEARS).fillna(0.08)
premium_rate        = premium_rate.reindex(YEARS).fillna(premium_rate.iloc[0])
ret_age_m           = ret_age_m.reindex(YEARS).fillna(ret_age_m.iloc[0])
ret_age_w           = ret_age_w.reindex(YEARS).fillna(ret_age_w.iloc[0])
pension_other       = pension_other.reindex(YEARS).fillna(0)

# ----------------------------
# 4. توابع تولید و بازده
# ----------------------------
def production_function(K, L, A=1.0):
    """تابع تولید کاب-داگلاس: Y = A * K^alpha * L^(1-alpha)"""
    # برای جلوگیری از تقسیم بر صفر
    K = max(K, 1e-9)
    L = max(L, 1e-9)
    return A * (K**ALPHA) * (L**(1-ALPHA))

def wage_rate_equil(K, L, A=1.0):
    """مزد تعادلی از مشتق تابع تولید نسبت به L"""
    K = max(K, 1e-9)
    L = max(L, 1e-9)
    return (1-ALPHA) * A * (K/L)**ALPHA

def capital_return_equil(K, L, A=1.0):
    """بازده سرمایه تعادلی از مشتق تابع تولید نسبت به K"""
    K = max(K, 1e-9)
    L = max(L, 1e-9)
    return ALPHA * A * (L/K)**(1-ALPHA)

# ----------------------------
# 5. قانون ساده سرمایه K_t
# ----------------------------
# K_t = THETA_K * (WageBill_{t-1})
# WageBill_{t} = Σ (Pop * Participation * Wage) برای گروه‌های شاغل
# اگر t-1 وجود نداشت (مثلاً سال اول)، از همان سال t استفاده می‌کنیم یا یک مقدار اولیه کوچک می‌گذاریم.

wage_bill_cache = {}

def compute_wage_bill(year):
    """صورت‌حساب دستمزد شاغلان در یک سال (برای دو گروه شاغل)"""
    bill = 0
    for cohort in ['15-29', '30-64']:
        for sex in SEXES:
            N = population.loc[year, f'{cohort}_{sex}']
            part = participation.loc[year, f'{cohort}_{sex}']
            w = wage.loc[year, f'{cohort}_{sex}']
            bill += N * part * w
    return bill

# ----------------------------
# 6. سناریوهای سیاستی
# ----------------------------
# نکته: ما در اینجا جمعیت را دست‌کاری نمی‌کنیم؛
# فقط تعیین می‌کنیم چه کسی حق دریافت مستمری دارد یا نرخ بیمه چقدر است.
# برای ساده‌سازی، گروه 65+ همیشه بازنشسته است.
# اگر سن بازنشستگی را بالا می‌بریم، فقط پرداخت مستمری به بخشی از گروه 30-64 قطع می‌شود.

# کمک: درصدی از گروه 30-64 که به‌دلیل اصلاح سن بازنشستگی هنوز بازنشسته نیستند
# (می‌توان پیچیده‌تر کرد، اما ساده نگه می‌داریم.)

def retired_share_in_30_64(year, ret_age):
    """
    سهم بازنشستگان در گروه 30-64 بر اساس سن بازنشستگی جدید.
    فرض می‌کنیم گروه 30-64 35 سال طول دارد (30 تا 64).
    اگر ret_age = 62 => فقط افراد 62 تا 64 (3 سال از 35 سال) بازنشسته می‌شوند => سهم ≈ 3/35
    این ساده‌سازی است برای جلوگیری از دستکاری جمعیت؛ همین کافی‌ست.
    """
    total_span = 35
    retired_span = max(0, 64 - ret_age + 1)  # تعداد سال‌های بازنشسته داخل این گروه
    return retired_span / total_span

# سناریو 1: پایه

def scenario_baseline(year):
    premium = premium_rate.iloc[0]  # نرخ بیمه ثابت بر اساس سال پایه
    retm = ret_age_m.iloc[0]
    retw = ret_age_w.iloc[0]
    return premium, retm, retw

# سناریو 2: اصلاح سن بازنشستگی

def scenario_retirement_reform(year):
    m0 = ret_age_m.iloc[0]
    w0 = ret_age_w.iloc[0]
    if year <= 1410:
        retm = m0 + (year-1400)*(65-m0)/10.0
        retw = w0 + (year-1400)*(60-w0)/10.0
    else:
        retm, retw = 65, 60
    premium = premium_rate.iloc[0]
    return premium, retm, retw

# سناریو 3: اصلاح نرخ بیمه

def scenario_premium_reform(year):
    base = premium_rate.iloc[0]  # مثلا 0.30
    target = 0.35
    if year <= 1410:
        premium = base + (year-1400)*(target - base)/10.0
    else:
        premium = target
    retm = ret_age_m.iloc[0]
    retw = ret_age_w.iloc[0]
    return premium, retm, retw

# سناریو 4: حمایت دولت (دولت کسری صندوق را می‌پردازد)

def scenario_govt_budget(year):
    premium = premium_rate.iloc[0]
    retm = ret_age_m.iloc[0]
    retw = ret_age_w.iloc[0]
    return premium, retm, retw

# ----------------------------
# 7. حلقه شبیه‌سازی اصلی
# ----------------------------

def run_simulation(scenario_func, with_govt=False):
    """
    اجرای مدل برای یک سناریو مشخص.
    اگر with_govt=True باشد، مازاد/کسری صندوق به بودجه دولت می‌رود و انباشته می‌شود.
    """
    results = []
    A_t = A_0
    govt_budget_balance = 0.0
    K_prev = 1e9  # مقدار اولیه سرمایه (دلخواه، فقط برای شروع)

    for year in YEARS:
        # 1) بهره‌وری کل عوامل
        A_t *= (1 + productivity_growth.get(year, 0))

        # 2) پارامترهای سناریو
        premium, retm, retw = scenario_func(year)

        # 3) نیروی کار فعال (گروه‌های 15-29 و 30-64)
        L_t = 0.0
        for cohort in ['15-29', '30-64']:
            for sex in SEXES:
                N = population.loc[year, f'{cohort}_{sex}']
                part = participation.loc[year, f'{cohort}_{sex}']
                L_t += N * part

        # 4) سرمایه K_t طبق قانون ساده
        # ابتدا WageBill سال قبل را محاسبه/بخوان. اگر سال قبل نیست، از همین سال استفاده کن.
        if year-1 in YEARS:
            bill_prev = wage_bill_cache.get(year-1, compute_wage_bill(year-1))
        else:
            bill_prev = compute_wage_bill(year)
        wage_bill_cache[year-1] = bill_prev
        K_t = THETA_K * bill_prev
        # اگر خواستی از K_prev هم استفاده کنی (انباشت سرمایه)، می‌توانی: K_t = (1-δ)*K_prev + THETA_K*bill_prev
        # ولی فعلاً ساده نگه‌می‌داریم.

        # 5) تولید و قیمت عوامل
        Y_t = production_function(K_t, L_t, A_t)
        w_t = wage_rate_equil(K_t, L_t, A_t)
        r_t = capital_return_equil(K_t, L_t, A_t)

        # 6) منابع صندوق (حق بیمه) فقط شاغلان 15-29 و 30-64
        premiums = 0.0
        for cohort in ['15-29', '30-64']:
            for sex in SEXES:
                N = population.loc[year, f'{cohort}_{sex}']
                part = participation.loc[year, f'{cohort}_{sex}']
                w_avg = wage.loc[year, f'{cohort}_{sex}']
                premiums += N * part * w_avg * premium

        # 7) مصارف صندوق (مستمری)
        pensions_total = 0.0
        # الف) گروه 65+
        for sex in SEXES:
            N = population.loc[year, f'65+_{sex}']
            pensions_total += N * pension.loc[year, f'65+_{sex}']
        # ب) بخشی از گروه 30-64 که به‌علت رسیدن به سن بازنشستگی، مستمری می‌گیرند (در سناریوی اصلاح سن بازنشستگی)
        share_m = retired_share_in_30_64(year, retm)
        share_w = retired_share_in_30_64(year, retw)
        # اگر سناریو پایه یا اصلاح نرخ بیمه است، این share در حد پایه خواهد بود (با retm,retw اولیه)
        for sex, share in zip(SEXES, [share_m, share_w]):
            N = population.loc[year, f'30-64_{sex}'] * share
            pensions_total += N * pension.loc[year, f'30-64_{sex}']

        # ج) هزینه‌های دیگر (اختیاری)
        pensions_total += pension_other.get(year, 0)

        # 8) تراز صندوق
        balance = premiums - pensions_total

        # 9) بودجه دولت (اختیاری)
        if with_govt:
            govt_budget_balance += balance

        # 10) ثبت نتایج
        results.append({
            'Year': year,
            'A': A_t,
            'K': K_t,
            'L': L_t,
            'Output': Y_t,
            'Wage_equil': w_t,
            'Return_equil': r_t,
            'Premiums': premiums,
            'Pensions': pensions_total,
            'Balance': balance,
            'Govt_Budget_Balance': govt_budget_balance if with_govt else np.nan,
            'PremiumRate': premium,
            'RetAgeM': retm,
            'RetAgeW': retw
        })

        K_prev = K_t  # اگر بخواهی مدل انباشتی سرمایه داشته باشی

    return pd.DataFrame(results).set_index('Year')

# ----------------------------
# 8. نمونه اجرای مدل (راهنما)
# ----------------------------
"""
برای اجرای سناریوها:

# سناریوی پایه:
df_base = run_simulation(scenario_baseline, with_govt=False)

# سناریوی اصلاح سن بازنشستگی:
df_ret  = run_simulation(scenario_retirement_reform, with_govt=False)

# سناریوی اصلاح نرخ بیمه:
df_prem = run_simulation(scenario_premium_reform, with_govt=False)

# سناریوی حمایت دولت از صندوق:
df_gov  = run_simulation(scenario_govt_budget, with_govt=True)

# ذخیره خروجی‌ها در اکسل یا CSV:
df_base.to_excel('results_base.xlsx')

#نکات:
#- اگر هر کدام از فایل‌های اکسل داده ناقص بود، قبل از اجرا پرش کن یا مقدار پیش‌فرض بده.
#- واحد پول و جمعیت را واحدسازی کن (مثلاً همه به ریال و نفر).
#- اگر خواستی سرمایه را دقیق‌تر کنی، می‌توانی معادله انباشت سرمایه یا پس‌انداز بهینه را اضافه کنی.
#"""

print("✅ مدل نهایی OLG سه‌دوره‌ای با تفکیک جنسیت و قانون ساده K بارگذاری شد. داده‌ها را وارد و سناریوها را اجرا کنید.")


#دستور اجرایی 5 THINKING
if __name__ == "__main__":
    import traceback
    print("▶ اجرای سناریوها...")

    try:
        # اجرای 4 سناریو (نام توابع را اگر در کدت فرق دارد، همین‌جا جایگزین کن)
        df_base = run_simulation(scenario_baseline,            with_govt=False)
        df_ret  = run_simulation(scenario_retirement_reform,   with_govt=False)
        df_prem = run_simulation(scenario_premium_reform,      with_govt=False)
        df_gov  = run_simulation(scenario_govt_budget,         with_govt=True)

        # ذخیره خروجی‌ها کنار فایل کد
        df_base.to_excel("results_base.xlsx")
        df_ret.to_excel("results_retirement.xlsx")
        df_prem.to_excel("results_premium.xlsx")
        df_gov.to_excel("results_govt.xlsx")

        # چند چک سریع روی ترمینال
        print("\n— چک سریع —")
        try:
            print("PremiumRate (1408..1412):")
            print(df_prem.loc[1408:1412, ["PremiumRate"]])
            print("Ret ages (1408..1412):")
            print(df_ret.loc[1408:1412, ["RetAgeM", "RetAgeW"]])
        except Exception:
            pass

        print("\n✅ نتایج ذخیره شد: results_*.xlsx")
    except Exception as e:
        print("❌ اجرای مدل با خطا مواجه شد:", e)
        traceback.print_exc()
