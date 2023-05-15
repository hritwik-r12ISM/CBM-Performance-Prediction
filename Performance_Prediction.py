import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import math
from PIL import Image
import math as m





st.set_page_config(page_title="CBM Performance Prediction", layout="wide")

st.title("Performance Prediction")
# st.markdown("Performance Prediction using python with following data: ")

st.sidebar.write("Reservoir parameters")


th=st.sidebar.number_input("Net Coal Thickness ft :",value=50.00)
Vl=st.sidebar.number_input("Langmuir Volume (scf/t) :",value=500.00)
Pl=st.sidebar.number_input("Langmuir Pressure (psia) :",value=400.00)
rhocoal=st.sidebar.number_input("Coal Density (g/cc) :",value=1.56)
Pi=st.sidebar.number_input("Initial Pressure (psia) :",value=1200.00)
T=st.sidebar.number_input("Reservoir Temperature (deg R) :",value=530)
phii=st.sidebar.number_input("Initial Porosity :",value=0.01)
cc=st.sidebar.number_input("Cleat Compressibilty (per psi) :",value=1.05E-04)
Swi=st.sidebar.number_input("Initial Water Saturation:",value=1.00)
Zw=st.sidebar.number_input("Water Compressibilty (per psi) :",value=3.20E-06)
winf=st.sidebar.number_input("Water Influx (bbls) :",value=0.00)
A=st.sidebar.number_input("Drainage Area (acres) :",value=1.60E+02)
Bwi=st.sidebar.number_input("Initial Formtion Volume Factor (Bwi) :",value=1.00)
td=st.sidebar.number_input("Desorption Time (days):",value=5.00)
Swc=st.sidebar.number_input("Connate Water Saturation (Swc) :",value=0.20)
Sgc=st.sidebar.number_input("Critical Gas Saturation (Sgc) :",value=0.04)
Krwe=st.sidebar.number_input("Endpoint water relative permeability (Krwe):",value=0.95)
Krge=st.sidebar.number_input("Endpoint gas relative permeability (Krge):",value=0.95)
nw=st.sidebar.number_input("Exponent for water (nw) :",value=2.9)
ng=st.sidebar.number_input("Exponent for gas (ng) :",value=1.90)
muw=st.sidebar.number_input("Water viscosity (cp) :",value=1.00)
K=st.sidebar.number_input("Permeability (K) :",value=100.00)
re=st.sidebar.number_input("Drainage Radius (re) :",value=1489.46)
rw=st.sidebar.number_input("Wellbore Radius (rw) :",value=0.3)
Skin=st.sidebar.number_input("Skin :",value=-2)
rcw=st.sidebar.number_input("Water Rate Constraint (bbl/day):",value=100.00)
BHPmin=st.sidebar.number_input("BHPmin :",value=1.00E+02)

# from math import e,log


import math as m

P = [Pi]
mu = []
Z = []
Bw = []
Sw = []
Sg = []
phi = []
delzp = []
Cg = []
Cf = cc
Bg = []
Cs = []
tau = [0]
t = [0]
Cstau = [0]
Ct = []
Krw = []
Krg = []
KrgKrw = []
Qg = [0]
Qwsc = []
Qt = []
lambdat = []
tp = [0]
tpavg = [0]
deltat = [0]
tpavgdelt = [0]
tpinteg = [0]
tpseudo = [0]
tpseudototal = [0]
Padiff = [0]
lambdatavg = [0]
lambdatdelPinteg = [0]
Ptemp = [0]
deltaP = [0]
Pawfdiff = [0]
Pawf = [0]
Wp = [0]
Gp = [0]
CumGp = [0]
Tc = 343.1
Pc = 667.8
Ma = 16
SgCH4 = 0.5537 #https://miningquiz.com/download/GasSpecificGravity.htm

cnt = 0

for i in range(0,3650,5):

  # Lee et al. Tarek Ahmed Page 67

  K = ((9.4 + (0.02 * Ma)) * (T ** 1.5))/(209 + (19 * Ma) + T)
  X = 3.5 + (986/T) +(0.01 * Ma)
  Y = 2.4 - (0.2 * X)

  mu1 = 0.0001 * K * m.exp(X * ((SgCH4 * 28.96/62.4) ** Y))
  mu.append(mu1)

  # Assuming pure methane

  Pr = P[cnt]/Pc
  Tr = T/Tc

  # Calculating Z by Ahmed (2017) Tarek Ahmed 5th Edition Page Number 42
  Z1 = 1.008505 + (0.04623 * (Pr/Tr)) + ((0.862707 * (Pr ** 1.368627))/(10 ** (0.636778 * Tr))) - ((2.324825 * Pr)/(10 ** (0.649787 * Tr)))
  Z.append(Z1)

  Bw1 = Bwi * (1 + Zw * (Pi - P[cnt]))
  Bw.append(Bw1)

  Sw1 = (Swi * (1 + Zw * (Pi - P[cnt])) + (Bw1 * (winf - Wp[cnt])/(7758.4 * A * th * phii)))/(m.exp(-1 * cc * (Pi - P[cnt])))
  Sw.append(Sw1)

  Sg1 = 1-Sw1
  Sg.append(Sg1)

  phi1 = phii * m.exp(-1 * cc * (Pi - P[cnt]))
  phi.append(phi1)

  delzp1 = 0.0000000000000046854 * (P[cnt]**3) + 0.000000000011922 * (P[cnt]**2) + 0.000000013576*P[cnt] - 0.0001389
  delzp.append(delzp1)

  Cg1 = (1/P[cnt]) - (delzp1/Z1)
  Cg.append(Cg1)

  Bg1 = 0.00503676 * Z1 * T/P[cnt]
  Bg.append(Bg1)

  Cs1 = Bg1 * 5.615 * rhocoal * (Vl/35.31) * Pl/(phii * m.pow((Pl + P[cnt]),2))
  Cs.append(Cs1)

  if t[cnt]>0:

    tau1 = 1/(1 - m.exp(-1 * t[cnt]/td))
    tau.append(tau1)

    Cstau1 = Cs1/tau1
    Cstau.append(Cstau1)

  Ct1 = Cf + (Cg1 * Sg1) + (Zw * Sw1) + Cstau[cnt]
  Ct.append(Ct1)

  if Sw1 <= Swc:
    Krw1 = 0
  else:
    if Sw1 >= (1-Sgc):
      Krw1 = Krwe
    else:
      Krw1 = Krwe * ((Sw1 - Swc)/(1 - Swc - Sgc) ** nw)
  Krw.append(Krw1)

  if Sg1 <= Sgc:
    Krg1 = 0
  else:
    if Sg1 >= (1 - Swc):
      Krg1 = Krge
    else:
      Krg1 = Krge * ((Sg1 - Sgc)/(1 - Swc - Sgc) ** ng)
  Krg.append(Krg1)

  KrgKrw.append(Krg1/Krw1)

  if cnt<=1:
    Qwsc1 = rcw
  else:
    if Pawf[cnt - 1] <= BHPmin:
      Qwsc1 = (0.00708 * K * Krw[cnt - 1] * th * (P[cnt - 1] - 100))/((m.log(re/rw) - 0.75 + Skin) * muw * Bw[cnt-1])
    else:
      Qwsc1 = rcw

  Qwsc.append(Qwsc1)

  if Qwsc1 == 0:
    lambdat1 = 0
  else:
    lambdat1 = ((K * Krw1/muw) + (K * Krg1/mu1))
  lambdat.append(lambdat1)

  if i==0:
    Qt.append(Qwsc1 * Bw1)

  if i == 3645:
    break

  Qg1 = (KrgKrw[cnt] * muw * Bw1 * Qwsc1)/(Bg1 * mu1)
  Qg.append(Qg1)

  Qt1 = (Qg1 * Bg1) + (Qwsc1 * Bw1)
  Qt.append(Qt1)

  t.append(t[cnt]+5)
  tp.append((Qt1 * lambdat1)/Ct1)
  cnt+=1

  tpavg.append((tp[cnt] + tp[cnt-1])/2)
  deltat.append(t[cnt]-t[cnt-1])
  tpavgdelt.append(tpavg[cnt] * deltat[cnt])
  tpinteg.append(tpavgdelt[cnt] + tpinteg[cnt-1])
  tpseudo.append((tpavgdelt[cnt]*Ct[cnt - 1])/(Qt[cnt - 1]*lambdat[cnt - 1]))
  tpseudototal.append(tpseudo[cnt] + tpseudototal[cnt - 1])
  Padiff.append((1.79 * Qt[cnt - 1] * tpseudo[cnt])/(m.pow(re,2) * th * phii * Ct[cnt - 1]))

  if cnt == 1:
    lambdatavg.append(lambdat[cnt - 1]/2)
  else:
    lambdatavg.append((lambdat[cnt - 1] + lambdat[cnt - 2])/2)

  lambdatdelPinteg.append(Padiff[cnt] * lambdat[cnt - 1])
  Ptemp.append(lambdatdelPinteg[cnt]/lambdatavg[cnt])

  if cnt == 1:
    deltaP.append(Padiff[cnt])
  else:
    deltaP.append(Ptemp[cnt] + deltaP[cnt - 1])

  P.append(Pi - deltaP[cnt])
  Pawfdiff.append((141.2 * Qt[cnt-1] * (m.log(re/rw) - 0.75 + Skin))/(lambdat[cnt - 1] * th))

  if P[cnt] - Pawfdiff[cnt] <= 100:
    Pawf.append(100)
  else:
    Pawf.append(P[cnt] - Pawfdiff[cnt])

  Wp.append(Qwsc[cnt - 1] * (t[cnt] - t[cnt-1]) + Wp[cnt - 1])
  Gp.append(Qg[cnt] * (t[cnt] - t[cnt-1]))
  CumGp.append(CumGp[cnt-1] + Gp[cnt])

# print(Gp)

# Calculation of Bulk density: General range: 1.25 g/cc to 1.70 g/cc. Increases with the rank of coal
# Reference: https://www.ias.ac.in/article/fulltext/jess/117/02/0121-0132

rhoash = 2.55
rhov = 1.29
rholi = 1.18
rhoin = 1.39
rhow = 1
a = 0.20465 #Average range 10.92% to 30.01%.
w = 0.0213 #Average range 1.28% to 2.98%
Vv = 0.72825 #Average Vitrinite 62.50 to83.15%
Vli = 0.01305 #Average Liptinite 0.66% to 3.09%
Vin = 0.2587 #Average range Inertinite 14.93 to 36.81%
rhoo = (rhov * Vv) + (rholi * Vli) + (rhoin * Vin)
den = ((1-a-w)/rhoo) + (a/rhoash) + (w/rhow)
rhobulk = 1/den


st.markdown(
    f"""
    * Ash density : {2.55}
    * Density of Vitrinite : {1.29}
    * Density of Liptinite : {1.18}
    * Density of Inertinite : {1.39}
    * Density of water : {1}
    * Ash Content (Average range 10.92% to 30.01%) : {0.20465}
    * Moisture Content (Average range 1.28% to 2.98%) : {0.0213}
    * Percentage Vitrinite (Average Vitrinite 62.50 to 83.15%) : {0.72825}
    * Percentage Liptinite (Average Liptinite 0.66% to 3.09%) : {0.01305}
    * Percentage Inertinite(#Average range Inertinite 14.93 to 36.81%) : {0.2587}
    * Bulk Density : {rhobulk}
    """)
# image = Image.open('bulk density.png')
# st.image(image, caption="Bulk Density Formula", width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
#
# image = Image.open('organic matter density.png')
# st.image(image, caption="Organic Matter Density", width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

data = {
    't' : t,
    'Qwsc' : Qwsc,
    'Wp' : Wp,
    'Gp' : Gp,
    'CumGp': CumGp,
    'Pressure': P,
    'viscosity': mu,
    'Z': Z,
    'Bw' : Bw,
    'Sw' : Sw,
    'Sg' : Sg,
    'phi' : phi,
    'delz/p' : delzp,
    'Cg' : Cg,
    'Bg' : Bg,
    'Cs' : Cs,
    'Sg' : Sg,
    'tau' : tau,
    'Cs/tau' : Cstau,
    'Ct' : Ct,
    'Krw' : Krw,
    'Krg' : Krg,
    'Krg/Krw' : KrgKrw,
    'Qg' : Qg,
    'Qt' : Qt,
    'lambdat' : lambdat,
    'tp' : tp,
    'tpavg' : tpavg,
    'deltat' : deltat,
    'tpavgdelt' : tpavgdelt,
    'tpinteg' : tpinteg,
    'tpseudo' : tpseudo,
    'tpseudototal' : tpseudototal,
    'Padiff' : Padiff,
    'lambdatavg' : lambdatavg,
    'lambdatdelPinteg' : lambdatdelPinteg,
    'Ptemp' : Ptemp,
    'deltaP' : deltaP,
    'Pawfdiff' : Pawfdiff,
    'Pawf' : Pawf,
    }
#st.write(data)
#@st.experimental_memo

st.write("Data Generated: ")

df = pd.DataFrame(data)
st.write(df)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Pressure"],
    y=df["Cs"],
    name="Cs vs Pressure"
))
fig.update_xaxes(minor=dict(ticks="inside", ticklen=6, showgrid=True))
fig.update_layout(title_text = "Plot of Cs (per psi ) vs Pressure (psi)")
fig.update_xaxes(title_text="Pressure (psi)")
fig.update_yaxes(title_text="Cs (1/psi)")
st.write(fig)


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["t"],
    y=df["Qg"],
    name="Qg vs t"
))
fig.update_xaxes(minor=dict(ticks="inside", ticklen=6, showgrid=True))
fig.update_layout(title_text = "Plot of Gas flow rate (scf/day) vs time (days)")
fig.update_xaxes(title_text="Time (days)")
fig.update_yaxes(title_text="Gas flow rate (scfd)")
st.write(fig)

st.write("From the graph, it can be seen that Gas flow rate is zero for some time initially since dewatering of coal bed methane is required to generate gas production. When the critical gas saturation is reached, the gas production starts and it increases with time till it reaches the maximum flow rate. Thereby, the gas flow rate decreases over time.")


ax = sns.lmplot(x='Gp', y='Pressure', data=df, scatter_kws={'color':'blue'}, fit_reg=False)
ax.set(xlabel='Gp (scf)', ylabel='Pressure (psi)')
st.write(ax)


# fig = go.Figure()

# fig.add_trace(go.Scatter(
#     x=df["Gp"],
#     y=df["Pressure"],
#     name="Pressure vs Gp"
# ))
# fig.update_xaxes(minor=dict(ticks="inside", ticklen=6, showgrid=True))
# fig.update_layout(title_text = "Plot of Pressure (psi) vs Gas Produced (scf)")
# fig.update_xaxes(title_text="Gp (Scf)")
# fig.update_yaxes(title_text="Pressure (psia)")
# st.write(fig)


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["CumGp"],
    y=df["Pressure"],
    name="Pressure vs CumGp"
))
fig.update_xaxes(minor=dict(ticks="inside", ticklen=6, showgrid=True))
fig.update_layout(title_text = "Plot of Pressure (psi) vs Cumulative Gas Produced (scf)")
fig.update_xaxes(title_text="Cumulative Gp (Scf)")
fig.update_yaxes(title_text="Pressure (psi)")
st.write(fig)

st.write("From the graph, we can conclude that when the pressure declines, the gas production increases till it reaches a maximum value, after which it starts to decrease with the decrease in pressure")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["t"],
    y=df["CumGp"],
    name="CumGp vs t"
))
fig.update_xaxes(minor=dict(ticks="inside", ticklen=6, showgrid=True))
fig.update_layout(title_text = "Plot of Cumulative Gas Produced (scf) vs Time (days)")
fig.update_xaxes(title_text="time (days)")
fig.update_yaxes(title_text="Cumulative Gas Produced (scf)")
st.write(fig)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Sw"],
    y=df["Krg/Krw"],
    name="Krg/Krw vs Sw"
))
fig.update_xaxes(minor=dict(ticks="inside", ticklen=6, showgrid=True))
fig.update_layout(title_text = "Plot of Krg/Krw vs Water Saturation")
fig.update_xaxes(title_text="Sw")
fig.update_yaxes(title_text="Krg/Krw")
st.write(fig)

st.write("With decrease in the water saturation, relative permeability to gas increases.")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Wp"],
    y=df["Gp"],
    name="Gp vs Wp"
))
fig.update_xaxes(minor=dict(ticks="inside", ticklen=6, showgrid=True))
fig.update_layout(title_text = "Plot of Gas Produced (scf) vs Water Produced (scf)")
fig.update_xaxes(title_text="Wp")
fig.update_yaxes(title_text="Gp")
st.write(fig)


st.title("Conventional Methods")

st.subheader("King P/Z* Approach:")

Zstar = []
PbyZstar = []

for i in range(0,730):
  deno = (1-(Cf * (Pi - P[i]))) * (1 - Sw[i]) + ((rhobulk * Bg[i])/phii) * ((Vl * P[i])/(Pl + P[i]))
  Zstar.append(Z[i]/deno)
  PbyZstar.append(P[i]/Zstar[i])

data1 = {
    'Zstar' : Zstar,
    'PbyZstar' : PbyZstar,
    'CumGp' : CumGp
    }

dff = pd.DataFrame(data1)
st.write(dff)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=dff["CumGp"],
    y=dff["PbyZstar"],
    name="PbyZstar vs CumGp"
))
fig.update_xaxes(minor=dict(ticks="inside", ticklen=6, showgrid=True))
fig.update_layout(title_text = "Plot of P/Z* vs Gas Produced (scf)")
fig.update_xaxes(title_text="Gas Produced (scf)")
fig.update_yaxes(title_text="P/Z*")
st.write(fig)

# Using polyfit() to generate coefficients
import numpy as np
p = np.polyfit(CumGp, PbyZstar, 1)  # Last argument is degree of polynomial

st.write("Coefficient Values: a and b")
st.write(p[0])
st.write(p[1])

# At y=0 or PbyZstar = 0, CumGp = -b/a
xvalue = -p[1]/p[0]
st.write("Gas in place (scf)")
st.write(xvalue)

st.subheader("Jensen and Smith Approach:")

Gadsorbed = []
Gpin = ((1.3597 * (10 ** -3)) * A * th * rhobulk) * ((Vl * Pi)/(Pl + Pi))
GpJS = []
yval = []
for i in range(0,730):
  Gadsorbed.append(((1.3597 * (10 ** -3)) * A * th * rhobulk) * ((Vl * P[i])/(Pl + P[i])))
  GpJS.append(Gpin - Gadsorbed[i])
  yval.append(P[i]/(Pl + P[i]))

data2 = {
    'Sw': Sw,
    'Gadsorbed' : Gadsorbed,
    'GpJS' : GpJS,
    'P/Pl+P' : yval
    }

dataframe = pd.DataFrame(data2)
st.write(dataframe)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=dataframe["GpJS"],
    y=dataframe["P/Pl+P"],
    name="P/Pl+P vs GpJS"
))
fig.update_xaxes(minor=dict(ticks="inside", ticklen=6, showgrid=True))
fig.update_layout(title_text = "Plot of P/PL+P vs Gas Produced (scf)")
fig.update_xaxes(title_text="Gas Produced (scf)")
fig.update_yaxes(title_text="P/PL+P")
st.write(fig)

# Using polyfit() to generate coefficients
p1 = np.polyfit(GpJS, yval, 1)  # Last argument is degree of polynomial

st.write("Coeficient values: a and b")
st.write(p1[0])
st.write(p1[1])

# At y=0 or P/Pl+P = 0, CumGp = -b/a
xvalue = -p1[1]/p1[0]
st.write("Gas in place (scf)")
st.write(xvalue*1000000)
