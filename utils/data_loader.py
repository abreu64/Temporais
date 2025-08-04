import pandas as pd
from datetime import datetime

def load_csv_data(uploaded_file):
    """Carrega e valida arquivo CSV de forma robusta"""
    try:
        # Tentar ler o CSV com diferentes codificações
        for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        # Verificar se o DataFrame foi carregado corretamente
        if df.empty:
            raise ValueError("O arquivo está vazio ou não pôde ser lido corretamente")
        
        # Verificar estrutura mínima
        if len(df.columns) < 2:
            raise ValueError("O arquivo deve conter pelo menos 2 colunas (uma de data e uma numérica)")
        
        # Identificar automaticamente a coluna de data
        date_col = None
        for col in df.columns:
            try:
                # Tentar converter para datetime
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='raise')
                date_col = col
                break
            except (ValueError, TypeError):
                continue
        
        if not date_col:
            raise ValueError("Nenhuma coluna de data válida encontrada (formato esperado: AAAA-MM-DD)")
        
        # Identificar coluna numérica
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            raise ValueError("Nenhuma coluna numérica encontrada")
        
        value_col = numeric_cols[0]  # Seleciona a primeira coluna numérica
        
        # Ordenar por data e remover duplicatas
        df = df.sort_values(date_col).drop_duplicates(subset=[date_col])
        
        # Verificar se há dados suficientes após limpeza
        if len(df) < 12:
            raise ValueError("Dados insuficientes após limpeza (mínimo 12 registros necessários)")
        
        return df, date_col, value_col
        
    except Exception as e:
        raise ValueError(f"Erro ao carregar dados: {str(e)}")