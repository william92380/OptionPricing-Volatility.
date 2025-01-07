import math
from scipy.stats import norm
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Fonction pour calculer le prix de l'option avec Black-Scholes
def prix_option_bs(S, K, r, q, sigma, T, option_type='call'):
    # Vérification des valeurs de S et K pour éviter les erreurs de domaine
    if S <= 0 or K <= 0:
        raise ValueError("S et K doivent être positifs et non nuls.")
    if T <= 0 or sigma <= 0:
        raise ValueError("T et sigma doivent être positifs et non nuls.")
    
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == 'call':
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("Le type d'option doit être 'call' ou 'put'")

# Fonction pour calculer la volatilité implicite avec la méthode de Newton-Raphson
def volatilite_implicite(S, K, r, q, T, prix_observe, option_type='call'):
    sigma = 0.5  # Volatilité initiale
    tolerance = 1e-5
    max_iterations = 100

    for _ in range(max_iterations):
        prix_calcule = prix_option_bs(S, K, r, q, sigma, T, option_type)
        diff = prix_calcule - prix_observe
        if abs(diff) < tolerance:
            return sigma

        # Calcul de Vega avec vérification que T et sigma sont non nuls
        if T == 0 or sigma == 0:
            print("T ou sigma égale à zéro, arrêt du calcul.")
            return None
        
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        vega = S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)
        
        if abs(vega) > 1e-8:  # Condition pour éviter la division par zéro
            sigma -= diff / vega
        else:
            print("Vega est trop proche de zéro, impossible de continuer l'itération.")
            return None

    print("Échec de convergence.")
    return None

# Récupération des données pour une option Apple (AAPL)
aapl = yf.Ticker("AAPL")
date_expiration = aapl.options[0]
options_chain = aapl.option_chain(date_expiration)

# Sélection d'une option avec un prix d'exercice proche du prix actuel
S = aapl.history(period="1d")['Close'].iloc[-1]  # Prix actuel du sous-jacent
selected_call = options_chain.calls.loc[(options_chain.calls['strike'] - S).abs().idxmin()]
selected_put = options_chain.puts.loc[(options_chain.puts['strike'] - S).abs().idxmin()]

K = selected_call['strike']
prix_observe_call = selected_call['lastPrice']
prix_observe_put = selected_put['lastPrice']
r = 0.05  # Taux d'intérêt sans risque
q = 0  # Taux de dividende
T = max((pd.to_datetime(date_expiration) - pd.Timestamp.today()).days / 365, 1 / 365)  # Valeur minimale de 1 jour

# Vérifier les valeurs de S, K, T, et sigma
print("S (Prix du sous-jacent):", S)
print("K (Prix d'exercice):", K)
print("T (Temps à l'expiration):", T)
print("sigma (Volatilité):", 0.5)

# Calcul du prix et de la volatilité implicite
call_price = prix_option_bs(S, K, r, q, 0.5, T, option_type='call')
put_price = prix_option_bs(S, K, r, q, 0.5, T, option_type='put')
vol_implicite_call = volatilite_implicite(S, K, r, q, T, prix_observe_call, option_type='call')

# Affichage des résultats
print("Prix théorique du Call :", call_price)
print("Prix théorique du Put :", put_price)
print("Volatilité implicite (Call) :", vol_implicite_call)
print("Prix observé pour le Call :", prix_observe_call)
print("Prix observé pour le Put :", prix_observe_put)

# Filtrage des options liquides
volume_seuil = 100
calls_liquides = options_chain.calls[options_chain.calls['volume'] > volume_seuil]
puts_liquides = options_chain.puts[options_chain.puts['volume'] > volume_seuil]

print("\nOptions Call liquides :")
print(calls_liquides)
print("\nOptions Put liquides :")
print(puts_liquides)

# Visualisation de la volatilité implicite en fonction du prix d'exercice
strikes = calls_liquides['strike']
volatilite_implicite = calls_liquides['impliedVolatility']

plt.figure(figsize=(10, 6))
plt.plot(strikes, volatilite_implicite, marker='o', linestyle='-')
plt.title("Volatilité Implicite en Fonction du Strike pour une Maturité Spécifique (Apple)")
plt.xlabel("Prix d'Exercice (Strike)")
plt.ylabel("Volatilité Implicite")
plt.grid(True)
plt.show()
