import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, roc_auc_score
from limpieza import cargar_y_resumen_inicial, construir_bitacora_y_limpiar, limpiar

# Crear carpeta para guardar figuras
os.makedirs("figuras", exist_ok=True)

# PARTE 0: PROCESOS SIMULADOS
def autofunc_rk(y, k):
    n = len(y)
    y_bar = np.mean(y)
    num = np.sum((y[k:] - y_bar) * (y[:-k] - y_bar))
    den = np.sum((y - y_bar)**2)
    return num / den

def simular_ar1(phi, n=300, burn=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    total_n = n + burn
    eps = rng.normal(0, 1, total_n)
    y = np.zeros(total_n)
    for t in range(1, total_n):
        y[t] = phi * y[t-1] + eps[t]
    return y[burn:]

def experimento_parte0_2():
    phis = [0.0, 0.5, 0.9, 1.0]
    num_replicas = 5000
    n = 300
    banda = 1.96 / np.sqrt(n)
    
    res_tabla = []
    
    for i, phi in enumerate(phis):
        rng = np.random.default_rng(2026 + i)
        r1_list, r5_list, rechazo_list, ols_list = [], [], [], []
        
        for _ in range(num_replicas):
            y = simular_ar1(phi, n=n, burn=200, rng=rng)
            r1 = autofunc_rk(y, 1)
            r5 = autofunc_rk(y, 5)
            
            # OLS estimate
            y_t = y[1:]
            y_t1 = y[:-1]
            phi_ols = np.sum(y_t * y_t1) / np.sum(y_t1**2)
            
            r1_list.append(r1)
            r5_list.append(r5)
            rechazo_list.append(1 if abs(r1) > banda else 0)
            ols_list.append(phi_ols)
            
        r1_arr, r5_arr = np.array(r1_list), np.array(r5_list)
        rech_arr, ols_arr = np.array(rechazo_list), np.array(ols_list)
        
        res_tabla.append({
            'phi': phi,
            'mean_r1': np.mean(r1_arr), 'se_r1': np.std(r1_arr, ddof=1)/np.sqrt(num_replicas),
            'mean_r5': np.mean(r5_arr), 'se_r5': np.std(r5_arr, ddof=1)/np.sqrt(num_replicas),
            'teorico_r5': phi**5,
            'tasa_rechazo': np.mean(rech_arr), 'se_rechazo': np.std(rech_arr, ddof=1)/np.sqrt(num_replicas),
            'mean_phi_ols': np.mean(ols_arr), 'se_phi_ols': np.std(ols_arr, ddof=1)/np.sqrt(num_replicas)
        })
    return pd.DataFrame(res_tabla)

# PARTE 1 - 2: LIMPIEZA Y CALENDARIO
print("Cargando datos...")
df_raw = cargar_y_resumen_inicial("Online Retail.xlsx")
df_limpio, bitacora = construir_bitacora_y_limpiar(df_raw)

# Serie diaria
df_limpio['TotalNeto'] = df_limpio['Quantity'] * df_limpio['UnitPrice']
df_limpio['Fecha'] = df_limpio['InvoiceDate'].dt.date

serie_diaria = df_limpio.groupby('Fecha')['TotalNeto'].sum().reset_index()
serie_diaria['Fecha'] = pd.to_datetime(serie_diaria['Fecha'])

# Calendario Completo
fecha_min, fecha_max = serie_diaria['Fecha'].min(), serie_diaria['Fecha'].max()
calendario = pd.date_range(start=fecha_min, end=fecha_max, freq='D')
df_calendario = pd.DataFrame({'Fecha': calendario}).merge(serie_diaria, on='Fecha', how='left')
df_calendario['DiaSemana'] = df_calendario['Fecha'].dt.day_name()

# Excluir domingos (cerrados) y último día incompleto
df_calendario['Abierto'] = df_calendario['Fecha'].dt.dayofweek != 6 # 6 = Domingo
df_calendario.loc[df_calendario['Fecha'] == fecha_max, 'Abierto'] = False

df_trabajo = df_calendario[df_calendario['Abierto']].copy().reset_index(drop=True)
df_trabajo['TotalNeto_Relleno'] = df_trabajo['TotalNeto'].fillna(0)

# Separación Entrenamiento y Prueba
f_train_end = pd.to_datetime('2011-09-30')
train_mask = df_trabajo['Fecha'] <= f_train_end
df_train = df_trabajo[train_mask].copy()
df_test = df_trabajo[~train_mask].copy()

# PARTE 6: ANÁLISIS RFM
def ejecutar_rfm(df_limpio):
    # Solo transacciones con CustomerID previo a 2011-10-01
    df_rfm_raw = df_limpio[
        (df_limpio['CustomerID'].notna()) & 
        (df_limpio['CustomerID'] != 'nan') & 
        (df_limpio['InvoiceDate'] < '2011-10-01')
    ].copy()
    
    f_corte = pd.to_datetime('2011-10-01')
    
    rfm = df_rfm_raw.groupby('CustomerID').agg(
        UltimaCompra=('InvoiceDate', 'max'),
        Frecuencia=('InvoiceNo', 'nunique'),
        Monto=('TotalNeto', 'sum')
    ).reset_index()
    
    rfm['Recencia'] = (f_corte - rfm['UltimaCompra']).dt.days
    
    # Filtrar Monto <= 0 para la puntuación, pero mantener según indicación
    # Desempate aleatorio con semilla 2026
    rng = np.random.default_rng(2026)
    rfm = rfm.sort_values('CustomerID').reset_index(drop=True)
    perm = rng.permutation(len(rfm))
    rfm = rfm.iloc[perm].reset_index(drop=True)
    
    rfm['R_score'] = pd.qcut(rfm['Recencia'].rank(method='first', ascending=False), 5, labels=[1,2,3,4,5])
    rfm['F_score'] = pd.qcut(rfm['Frecuencia'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M_score'] = pd.qcut(rfm['Monto'].rank(method='first'), 5, labels=[1,2,3,4,5])
    
    # Compró en periodo de prueba (2011-10-01 a 2011-12-08)
    df_post = df_limpio[
        (df_limpio['CustomerID'].notna()) & 
        (df_limpio['InvoiceDate'] >= '2011-10-01') & 
        (df_limpio['InvoiceDate'] <= '2011-12-08')
    ]
    clientes_post = set(df_post['CustomerID'].unique())
    rfm['Compro'] = rfm['CustomerID'].apply(lambda x: 1 if x in clientes_post else 0)
    
    auc_r = roc_auc_score(rfm['Compro'], -rfm['Recencia'])
    auc_f = roc_auc_score(rfm['Compro'], rfm['Frecuencia'])
    auc_m = roc_auc_score(rfm['Compro'], rfm['Monto'])
    
    rfm['Suma_Score'] = rfm['R_score'].astype(int) + rfm['F_score'].astype(int) + rfm['M_score'].astype(int)
    auc_suma = roc_auc_score(rfm['Compro'], rfm['Suma_Score'])
    
    return rfm, {'AUC_R': auc_r, 'AUC_F': auc_f, 'AUC_M': auc_m, 'AUC_Suma': auc_suma}

if __name__ == "__main__":
    print("Ejecutando simulación Parte 0...")
    df_p0 = experimento_parte0_2()
    print(df_p0)
    
    print("Ejecutando RFM Parte 6...")
    rfm_df, aucs = ejecutar_rfm(df_limpio)
    print("Resultados AUC RFM:", aucs)
    
    print("Proceso completado exitosamente.")
