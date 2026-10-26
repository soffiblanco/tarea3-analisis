import pandas as pd
import numpy as np

def cargar_y_resumen_inicial(filepath):
    """
    Carga el dataset Online Retail especificando tipos de datos correctos para InvoiceNo,
    StockCode y CustomerID como texto.
    """
    df = pd.read_excel(
        filepath,
        dtype={
            'InvoiceNo': str,
            'StockCode': str,
            'CustomerID': str
        }
    )
    # Limpieza básica de espacios
    df['InvoiceNo'] = df['InvoiceNo'].astype(str).str.strip()
    df['StockCode'] = df['StockCode'].astype(str).str.strip()
    df['CustomerID'] = df['CustomerID'].astype(str).str.strip()
    return df

def construir_bitacora_y_limpiar(df):
    """
    Aplica secuencialmente las reglas de limpieza y genera la bitácora requerida.
    """
    bitacora = []
    df_actual = df.copy()
    
    # 1. Filas duplicadas exactas
    dup_mask = df_actual.duplicated()
    num_dup = dup_mask.sum()
    monto_dup = (df_actual.loc[dup_mask, 'Quantity'] * df_actual.loc[dup_mask, 'UnitPrice']).sum()
    bitacora.append({'Regla': '1. Eliminar duplicados exactos', 'Filas Eliminadas': num_dup, 'Monto (£)': monto_dup})
    df_actual = df_actual[~dup_mask]

    # 2. Facturas que no empiezan con un dígito (Cancelaciones C, A)
    non_digit_mask = ~df_actual['InvoiceNo'].str[0].str.isdigit()
    num_non_digit = non_digit_mask.sum()
    monto_non_digit = (df_actual.loc[non_digit_mask, 'Quantity'] * df_actual.loc[non_digit_mask, 'UnitPrice']).sum()
    bitacora.append({'Regla': '2. Cancelaciones / Facturas no numéricas', 'Filas Eliminadas': num_non_digit, 'Monto (£)': monto_non_digit})
    df_actual = df_actual[~non_digit_mask]

    # 3. StockCodes del sistema (POST, D, BANK CHARGES, M, CRUK)
    # Se filtran códigos que no coinciden con patrón estándar de productos
    non_product_codes = ['POST', 'D', 'M', 'PADS', 'DOT', 'CRUK', 'BANK CHARGES', 'AMAZONFEE']
    stock_mask = df_actual['StockCode'].isin(non_product_codes) | df_actual['StockCode'].str.startswith('gift_')
    num_stock = stock_mask.sum()
    monto_stock = (df_actual.loc[stock_mask, 'Quantity'] * df_actual.loc[stock_mask, 'UnitPrice']).sum()
    bitacora.append({'Regla': '3. Códigos del sistema / No productos', 'Filas Eliminadas': num_stock, 'Monto (£)': monto_stock})
    df_actual = df_actual[~stock_mask]

    # 4. Precios menores o iguales a cero
    price_mask = df_actual['UnitPrice'] <= 0
    num_price = price_mask.sum()
    monto_price = (df_actual.loc[price_mask, 'Quantity'] * df_actual.loc[price_mask, 'UnitPrice']).sum()
    bitacora.append({'Regla': '4. Precios Unitarios <= 0', 'Filas Eliminadas': num_price, 'Monto (£)': monto_price})
    df_actual = df_actual[~price_mask]

    # 5. Cantidades negativas en ventas válidas
    qty_mask = df_actual['Quantity'] <= 0
    num_qty = qty_mask.sum()
    monto_qty = (df_actual.loc[qty_mask, 'Quantity'] * df_actual.loc[qty_mask, 'UnitPrice']).sum()
    bitacora.append({'Regla': '5. Cantidades <= 0 restantes', 'Filas Eliminadas': num_qty, 'Monto (£)': monto_qty})
    df_actual = df_actual[~qty_mask]

    bitacora_df = pd.DataFrame(bitacora)
    return df_actual, bitacora_df

def limpiar(df):
    """
    Función única de limpieza reinterpretable que devuelve el DataFrame limpio.
    """
    df_limpio, _ = construir_bitacora_y_limpiar(df)
    return df_limpio
