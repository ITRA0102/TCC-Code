import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tkinter as tk
import plotly.express as px
import plotly.graph_objects as go
from tkinter import filedialog
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import os

def selecionar_arquivos():
    """Abre uma janela para selecionar arquivos CSV e retorna uma lista de caminhos dos arquivos selecionados."""
    root = tk.Tk()
    root.withdraw()  # Ocultar a janela principal
    arquivos_selecionados = filedialog.askopenfilenames(
        title="Selecionar arquivos CSV",
        filetypes=(("CSV files", "*.csv"), ("all files", "*.*")),
        multiple=False
    )
    root.destroy()
    return arquivos_selecionados

def carregar_dados(arquivo):
    """Carrega e processa dados do arquivo CSV."""
    df = pd.read_csv(arquivo, delimiter=';', header=None, skiprows=1, names=['Região', 'Descrição', 'Data_Hora', 'Valor'])
    df.dropna(subset=['Valor'], inplace=True)
    df['Data_Hora'] = pd.to_datetime(df['Data_Hora'], format='%d/%m/%Y %H:%M')
    df['Valor'] = df['Valor'].replace(r'[\.,]', '', regex=True).astype(float)
    df_se = df[df['Região'] == 'SE'].sort_values(by='Data_Hora')
    return df_se

def media_mensal(df):
    """Calcula a média mensal e traduz os meses para o português."""
    df['Mês'] = df['Data_Hora'].dt.month_name()
    media_mensal = df.groupby('Mês')['Valor'].mean().reset_index()
    meses_portugues = {
        'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março', 'April': 'Abril',
        'May': 'Maio', 'June': 'Junho', 'July': 'Julho', 'August': 'Agosto',
        'September': 'Setembro', 'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'
    }
    media_mensal['Mês'] = media_mensal['Mês'].map(meses_portugues)
    ordem_meses = list(meses_portugues.values())
    media_mensal['Mês'] = pd.Categorical(media_mensal['Mês'], categories=ordem_meses, ordered=True)
    media_mensal = media_mensal.sort_values('Mês').reset_index(drop=True)
    return media_mensal

def regressao_linear_mensal(df):
    """Realiza a regressão linear para cada mês e retorna previsões."""
    previsoes = []
    for mes in range(1, 13):
        df_mes = df[df['Data_Hora'].dt.month == mes]
        if len(df_mes) < 2:
            continue
        X = pd.to_numeric(df_mes['Data_Hora']).values.reshape(-1, 1)
        y = df_mes['Valor'].values
        X_treino, X_teste, y_treino, y_teste = (X, X, y, y) if len(df_mes) == 2 else train_test_split(X, y, test_size=0.2, random_state=42)
        modelo_regressao_linear = LinearRegression()
        modelo_regressao_linear.fit(X_treino, y_treino)
        datas_futuras = pd.date_range(start=df_mes['Data_Hora'].min(), end=df_mes['Data_Hora'].max(), freq='h')
        datas_futuras_df = pd.DataFrame({'Data_Hora': datas_futuras})
        X_futuro = pd.to_numeric(datas_futuras_df['Data_Hora']).values.reshape(-1, 1)
        previsao_regressao_linear = modelo_regressao_linear.predict(X_futuro)
        previsoes.append(pd.DataFrame({'Data_Hora': datas_futuras, 'Valor': previsao_regressao_linear}))
    return previsoes

def comparar_media_regressao_linear(df, previsoes):
    """Compara a média original com a média da regressão linear e calcula o erro médio relativo."""
    comparacao = []
    meses_portugues = {
        'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março', 'April': 'Abril',
        'May': 'Maio', 'June': 'Junho', 'July': 'Julho', 'August': 'Agosto',
        'September': 'Setembro', 'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'
    }
    for previsao in previsoes:
        mes_ingles = previsao['Data_Hora'].iloc[0].month_name()
        mes_portugues = meses_portugues.get(mes_ingles, mes_ingles)
        media_original = df[df['Data_Hora'].dt.month == previsao['Data_Hora'].iloc[0].month]['Valor'].mean()
        media_regressao_linear = previsao['Valor'].mean()
        comparacao.append({
            'Mês': mes_portugues,
            'Média Original': media_original,
            'Média Regressão Linear': media_regressao_linear
        })
    comparacao_df = pd.DataFrame(comparacao)
    comparacao_df['Erro Médio Relativo (%)'] = (
        abs(comparacao_df['Média Original'] - comparacao_df['Média Regressão Linear']) / 
        comparacao_df['Média Original'] * 100
    )
    return comparacao_df

def estimar_demanda_proximo_ano(df, modelo_regressao_linear):
    """Estima a demanda para o próximo ano usando o modelo de regressão linear."""
    ultima_data = df['Data_Hora'].max()
    datas_futuras = pd.date_range(start=ultima_data, periods=365, freq='D')
    X_futuro = pd.to_numeric(datas_futuras).values.reshape(-1, 1)
    previsao_proximo_ano = modelo_regressao_linear.predict(X_futuro)
    return pd.DataFrame({'Data_Hora': datas_futuras, 'Valor': previsao_proximo_ano})

def media_demanda_proximo_ano(previsao_proximo_ano_df):
    """Calcula a média da demanda do próximo ano para cada mês e traduz os meses para o português."""
    previsao_proximo_ano_df['Mês'] = previsao_proximo_ano_df['Data_Hora'].dt.month_name()
    media_demanda_proximo_ano = previsao_proximo_ano_df.groupby('Mês')['Valor'].mean().reset_index()
    meses_portugues = {
        'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março', 'April': 'Abril',
        'May': 'Maio', 'June': 'Junho', 'July': 'Julho', 'August': 'Agosto',
        'September': 'Setembro', 'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'
    }
    media_demanda_proximo_ano['Mês'] = media_demanda_proximo_ano['Mês'].map(meses_portugues)
    ordem_meses = list(meses_portugues.values())
    media_demanda_proximo_ano['Mês'] = pd.Categorical(media_demanda_proximo_ano['Mês'], categories=ordem_meses, ordered=True)
    media_demanda_proximo_ano = media_demanda_proximo_ano.sort_values('Mês').reset_index(drop=True)
    return media_demanda_proximo_ano

def criar_grafico_media_mensal(media_mensal_df, nome_arquivo):
    """Cria e salva o gráfico da média mensal com fundo transparente e fonte Arial."""
    fig = px.line(media_mensal_df, x='Mês', y='Valor', title='Média Mensal (W)', markers=True)
    fig.update_yaxes(title_text='Valor (W)')
    
    fig.update_layout(
        title_font=dict(family="Arial", size=18),
        xaxis_title_font=dict(family="Arial", size=14),
        yaxis_title_font=dict(family="Arial", size=14),
        font=dict(family="Arial", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.write_image(nome_arquivo)

def criar_grafico_comparacao(comparacao_df, nome_arquivo):
    """Cria e salva o gráfico da comparação entre a média original e a média da regressão linear."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comparacao_df['Mês'], y=comparacao_df['Média Original'], mode='lines+markers', name='Média original do ano'))
    fig.add_trace(go.Scatter(x=comparacao_df['Mês'], y=comparacao_df['Média Regressão Linear'], mode='lines+markers', name='Média regressão linear'))
    
    # Calcular os limites desejados para o eixo Y
    y_min = comparacao_df[['Média Original', 'Média Regressão Linear']].min().min() * 0.9  # 10% abaixo do valor mínimo
    y_max = comparacao_df[['Média Original', 'Média Regressão Linear']].max().max() * 1.1  # 10% acima do valor máximo
    
    # Atualizar o layout para mover a legenda para baixo do gráfico e definir fundo transparente
    fig.update_layout(
        title='Comparação entre média original do ano e média da regressão linear (W)',
        yaxis_title='Valor (W)',
        xaxis_title='Mês',
        legend=dict(
            orientation='h',  # Orientação horizontal
            yanchor='top',  # Âncoras verticais
            y=-0.4,  # Distância abaixo do gráfico
            xanchor='center',
            x=0.5
        ),
        margin=dict(b=100),  # Adicionar margem inferior para acomodar a legenda
        plot_bgcolor='rgba(0,0,0,0)',  # Fundo do gráfico transparente
        paper_bgcolor='rgba(0,0,0,0)',  # Fundo do papel transparente
        title_x=0.5,  # Centraliza o título horizontalmente
        yaxis=dict(range=[y_min, y_max])  # Definir o intervalo do eixo Y
    )
    
    fig.write_image(nome_arquivo, engine='kaleido')

def criar_grafico_previsao(previsao_proximo_ano_df, nome_arquivo):
    """Cria e salva o gráfico da previsão de demanda para o próximo ano com fundo transparente e fonte Arial."""
    fig = px.line(previsao_proximo_ano_df, x='Data_Hora', y='Valor', title='Previsão de demanda para o Próximo Ano (W)', markers=True)
    fig.update_yaxes(title_text='Valor (GW)')
    
    # Calcular os limites desejados para o eixo Y
    y_min = previsao_proximo_ano_df['Valor'].min() * 0.9  # 10% abaixo do valor mínimo
    y_max = previsao_proximo_ano_df['Valor'].max() * 1.1  # 10% acima do valor máximo
    
    fig.update_layout(
        title_font=dict(family="Arial", size=18),
        xaxis_title_font=dict(family="Arial", size=14),
        yaxis_title_font=dict(family="Arial", size=14),
        font=dict(family="Arial", size=12),
        plot_bgcolor='rgba(0,0,0,0)',  # Fundo do gráfico transparente
        paper_bgcolor='rgba(0,0,0,0)',  # Fundo do papel transparente
        title_x=0.5,  # Centraliza o título horizontalmente
        yaxis=dict(range=[y_min, y_max])  # Definir o intervalo do eixo Y
    )
    
    fig.write_image(nome_arquivo, engine='kaleido')

def criar_grafico_media_demanda(media_demanda_proximo_ano_resultado, nome_arquivo):
    """Cria e salva o gráfico da média da demanda do próximo ano com fundo transparente e fonte Arial."""
    fig = px.line(media_demanda_proximo_ano_resultado, x='Mês', y='Valor', title='Média mensal de demanda para o próximo Ano (W)', markers=True)
    fig.update_yaxes(title_text='Valor (W)')
    
    fig.update_layout(
        title_font=dict(family="Arial", size=18),
        xaxis_title_font=dict(family="Arial", size=14),
        yaxis_title_font=dict(family="Arial", size=14),
        font=dict(family="Arial", size=12),
        plot_bgcolor='rgba(0,0,0,0)',  # Fundo do gráfico transparente
        paper_bgcolor='rgba(0,0,0,0)',  # Fundo do papel transparente
        title_x=0.5  # Centraliza o título horizontalmente
    )
    
    fig.write_image(nome_arquivo, engine='kaleido')


# Outras funções permanecem as mesmas

def exportar_para_excel(media_mensal_df, comparacao_df, media_demanda_proximo_ano_resultado, previsao_proximo_ano_df):
    """Exporta os dados e gráficos para um arquivo Excel."""
    if not os.path.exists('dados exportados'):
        os.makedirs('dados exportados')
    
    with pd.ExcelWriter('dados exportados/relatorio.xlsx', engine='xlsxwriter') as writer:
        media_mensal_df.to_excel(writer, sheet_name='Média Mensal', index=False)
        comparacao_df.to_excel(writer, sheet_name='Comparação Média', index=False)
        media_demanda_proximo_ano_resultado.to_excel(writer, sheet_name='Média Demanda Ano', index=False)
        previsao_proximo_ano_df.to_excel(writer, sheet_name='Previsão Próximo Ano', index=False)

        workbook  = writer.book
        worksheet = writer.sheets['Média Mensal']
        criar_grafico_media_mensal(media_mensal_df, 'dados exportados/grafico_media_mensal.png')
        worksheet.insert_image('E2', 'dados exportados/grafico_media_mensal.png')

        worksheet = writer.sheets['Comparação Média']
        criar_grafico_comparacao(comparacao_df, 'dados exportados/grafico_comparacao.png')
        worksheet.insert_image('E2', 'dados exportados/grafico_comparacao.png')

        worksheet = writer.sheets['Previsão Próximo Ano']
        criar_grafico_previsao(previsao_proximo_ano_df, 'dados exportados/grafico_previsao_proximo_ano.png')
        worksheet.insert_image('E2', 'dados exportados/grafico_previsao_proximo_ano.png')

        worksheet = writer.sheets['Média Demanda Ano']
        criar_grafico_media_demanda(media_demanda_proximo_ano_resultado, 'dados exportados/grafico_media_demanda_proximo_ano.png')
        worksheet.insert_image('E2', 'dados exportados/grafico_media_demanda_proximo_ano.png')

def main():
    arquivos = selecionar_arquivos()
    if not arquivos:
        print("Nenhum arquivo selecionado.")
        return

    df = carregar_dados(arquivos[0])
    media_mensal_df = media_mensal(df)
    previsoes = regressao_linear_mensal(df)
    comparacao_df = comparar_media_regressao_linear(df, previsoes)

    # Assume que o modelo de regressão linear é treinado com os dados do primeiro mês
    X = pd.to_numeric(df['Data_Hora']).values.reshape(-1, 1)
    y = df['Valor'].values
    modelo_regressao_linear = LinearRegression().fit(X, y)
    
    previsao_proximo_ano_df = estimar_demanda_proximo_ano(df, modelo_regressao_linear)
    media_demanda_proximo_ano_resultado = media_demanda_proximo_ano(previsao_proximo_ano_df)
    
    exportar_para_excel(media_mensal_df, comparacao_df, media_demanda_proximo_ano_resultado, previsao_proximo_ano_df)
    print("Relatório exportado com sucesso!")

if __name__ == "__main__":
    main()
