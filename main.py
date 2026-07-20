import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Iicializar o app dash
app = dash.Dash(__name__)

# Carregando o dataset de vendas
df = pd.read_csv('vendas.csv')

# Classe para estrutura de analise de dados
class AnalisadorDeVendas:
    def __init__(self, dados):
        ''' Inicializa a classe com o dataframe de vendas '''
        self.dados = dados
        self.limpar_dados()

    def limpar_dados(self):
        ''' Limpeza e preparação dos dados para análise '''
        self.dados['data'] = pd.to_datetime(self.dados['data']) # Converte data para Datetime
        self.dados['valor'] = self.dados['valor'].replace({',':'.'}, regex=True).astype(float) # Corrige os valores monetários
        self.dados.dropna(subset=['produto', 'valor'], inplace=True) # Remove dados ausente em colunas importantes
        self.dados['mes'] = self.dados['data'].dt.month # adiciona a coluna mes
        self.dados['ano'] = self.dados['data'].dt.year # adiciona a coluna ano
        self.dados['dia'] = self.dados['data'].dt.day # adiciona a coluna dia
        self.dados['dia_da_semana'] = self.dados['data'].dt.weekday # adiciona a coluna dia da semana (0 = segunda... 6 domingo)

    def analise_vendas_por_produto(self, produtos_filtrados):
        ''' Retorna o grafico de vendas por produto '''
        df_produtos = self.dados[self.dados['produto'].isin(produtos_filtrados)]
        df_produto = df_produtos.groupby('produto')['valor'].sum().reset_index().sort_values(by = 'valor', ascending = False)
        fig = px.bar(df_produto, x = 'produto', y = 'valor', title = 'Vendas por Produto', color = 'valor')
        return fig

    def analise_vendas_por_regiao(self, regioes_filtradas):
        ''' Retorna o grafico de vendas totais por regiao '''
        df_regioes = self.dados[self.dados['regiao'].isin(regioes_filtradas)]
        df_regiao = df_regioes.groupby('regiao')['valor'].sum().reset_index().sort_values(by = 'valor', ascending = False)
        fig = px.pie(df_regiao, names = 'regiao', values = 'valor', title = 'Vendas por Região', color = 'valor')
        return fig
    
    def analise_vendas_mensais(self, ano_filtrado):
        ''' Retorna o grafico de vendas por mes (com linha de tendencia) '''
        df_mes = self.dados[self.dados['ano'] == ano_filtrado]
        df_mes = df_mes.groupby(['ano', 'mes'])['valor'].sum().reset_index()
        fig = px.line(df_mes, x = 'mes', y = 'valor', title = f'Vendas mensais - {ano_filtrado}', color='ano', markers=True, line_shape='spline')
        return fig
    
    def analise_vendas_diarias(self, data_inicio, data_fim):
        ''' Retorna o grafico de vendas diarias ao longo do intervalo de tempo selecionado '''

        df_dia = self.dados[(self.dados['data'] >= data_inicio) & (self.dados['data'] <= data_fim)]
        df_dia = df_dia.groupby('data')['valor'].sum().reset_index()
        fig = px.line(df_dia, x = 'data', y = 'valor', title = 'Vendas Diarias', markers = True)
        return fig
    
    def analise_vendas_por_dia_da_semana(self):
        ''' Retorna o grafico de vendas por dia da semana (analisa o impacto do dia) '''
        df_dia_semana = self.dados.groupby('dia_da_semana')['valor'].sum().reset_index()
        df_dia_semana['dia_da_semana'] = df_dia_semana['dia_da_semana'].map({
            0:'Segunda', 1:'Terça', 2:'Quarta', 3:'Quinta', 4:'Sexta', 5:'Sabado', 6:'Domingo'
        })
        fig = px.bar(df_dia_semana, x = 'dia_da_semana', y = 'valor', title = 'Vendas por dia da semana', color = 'valor')
        return fig

    def analise_outliers(self):
        ''' Identifica os outliers com base em intervalo interquartil '''
        q1 = self.dados['valor'].quantile(0.25)
        q3 = self.dados['valor'].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        outliers = self.dados[(self.dados['valor'] < limite_inferior) | (self.dados['valor'] > limite_superior)]
        fig = px.scatter(outliers, x = 'data', y = 'valor', title = 'Outliers de Vendas')
        return fig

    def distribuicao_vendas(self):
        ''' Retorna o grafico de distribuicao de vendas utilizando o plotly '''
        fig = px.histogram(self.dados, x = 'valor', nbins = 30, title = 'Distribuição de vendas', color = 'valor')
        return fig
    
    def analise_media_desvio(self):
        ''' Calculo de media e desvio padrao'''
        media = self.dados['valor'].mean()
        desvio = self.dados['valor'].std()
        return media, desvio
    
    def vendas_acumuladas(self):
        ''' Calcula as vendas acumuladas ao longo do tempo com insights estatisticos'''
        df_acumulado = self.dados.groupby('data')['valor'].sum().cumsum().reset_index()

        #Calculos adicionais para enriquece a analise

        df_acumulado['media_movel_7'] = df_acumulado['valor'].rolling(window=7).mean() # media movel de 7 dias
        df_acumulado['media_padrao_7'] = df_acumulado['valor'].rolling(window=7).std() # desvio padrao de 7 dias
        df_acumulado['crescimento_percentual'] = df_acumulado['valor'].pct_change() * 100 # o pct_change pega cada valor, olha o anterior e calcula a variação percentual entre eles, é como automatizar toda a coluna de uma vez
        df_acumulado['max_valor'] = df_acumulado['valor'].expanding().max()
        df_acumulado['min_valor'] = df_acumulado['valor'].expanding().min()

        #Criação dográfico
        fig = px.line(
            df_acumulado,
            x = 'data',
            y = ['valor', 'media_movel_7', 'max_valor', 'min_valor'],
            title = 'Vendas acumuladas ao longo do tempo com insights estatisticos',
            labels = {'valor':'Vendas Acumuladas', 'media_movel_7':'Media Movel (7 Dias)', 'max valor':'Maximo Acumulado', 'min_valor':'Minimo Acumulado'},
            markers = True
        )
        # adicionando o crescimento percentual ao graico com uma linha de anotações
        fig.add_trace(go.Scatter(
            x = df_acumulado['data'],
            y = df_acumulado['crescimento_percentual'],
            mode = 'lines+markers',
            name = 'Crescimento Percentual',
            line = dict(color='orange', width=2, dash='dot'),
            yaxis = 'y2'
        ))
        #formatação do grafico
        fig.update_layout(
            title_font = dict(size = 20, family = 'Poppins', color = '#2980b9'),
            plot_bgcolor = '#34495e',
            paper_bgcolor = '#2c3e50',
            font = dict(color='#ecf0f1', family='Roboto'),
            xaxis = dict(
                title = 'Data',
                tickformat = '%Y-%m-%d',
                showgrid = True,
                gridcolor = '#7f8c8d',
                tickangle = 45
            ),
            yaxis = dict(
                title = 'Vendas Acumuladas',
                showgrid = True,
                gridcolor = '#7f8c8d',
            ),
            yaxis2 = dict(
                title = 'Crescimento Percentual (%)',
                overlaying = 'y',
                side = 'right',
                showgrid = False,
                tickformat = '.1f'
            ),
            legend = dict(
                title = 'Metricas',
                orientation = 'h',
                yanchor = 'bottom',
                y = 1.1,
                xanchor = 'center',
                x = 0.5
            ),
            hovermode = 'x unified',
            autosize = True,
            margin = dict(t=50, b=50, l=40, r=40),
            shapes = [
                dict(
                    type = 'line',
                    x0 = df_acumulado['data'].min(),
                    x1 = df_acumulado['data'].max(),
                    y0 = df_acumulado['max_valor'].max(),
                    y1 = df_acumulado['max_valor'].max(),
                    line = dict(color = 'red', width = 2, dash = 'dash'),
                    name = 'Maximo Historico'
                ),
                dict(
                    type = 'line',
                    x0 = df_acumulado['data'].min(),
                    x1 = df_acumulado['data'].max(),
                    y0 = df_acumulado['min_valor'].min(),
                    y1 = df_acumulado['min_valor'].min(),
                    line = dict(color = 'green', width = 2, dash = 'dash'),
                    name = 'Minimo Historico'
                )
                
            ]
        )
        return fig

# Instanciando o objeto de analise de vendas
analise = AnalisadorDeVendas(df)

#Layout do app Dash
app.layout = html.Div([
    html.H1("Dashboards de Analise de Vendas", style = {'textAlign':'center'}),
    html.Div([
        html.Label("Selecione dos produtos!"),
        dcc.Dropdown(
         id = 'produto-dropdown',
            options = [{'label': produto, 'value': produto} for produto in df['produto'].unique()],
            multi = True,
            value = df['produto'].unique().tolist(),
            style = {'width': '48%'}
        ),
        html.Label('Selecione as Regiões:'),
        dcc.Dropdown(
            id = 'regiao-dropdown',
            options = [{'label': regiao, 'value': regiao} for regiao in df['regiao'].unique()],
            multi = True,
            value = df['regiao'].unique().tolist(),
            style = {'width': '48%'}
        ),
        html.Label('Selecione o Ano:'),
        dcc.Dropdown(
            id = 'ano-dropdown',
            options = [{'label': ano, 'value': ano} for ano in df['ano'].unique()],
            value = df['ano'].unique().min(),
            style = {'width': '48%'}
        ),
        html.Label('Selecione o intervalo de datas:'),
        dcc.DatePickerRange(
            id = 'date-picker-range',
            start_date = df['data'].min().date(),
            end_date = df['data'].max().date(),
            display_format = 'YYYY-MM-DD',
            style = {'width': '48%'}
        )
    ], style = {'padding': '20px'}),

    # graficos!!!!
        html.Div([
        dcc.Graph(id = 'grafico-produto'),
        dcc.Graph(id = 'grafico-regiao'),
        dcc.Graph(id = 'grafico-mensal'),
        dcc.Graph(id = 'grafico-diario'),
        dcc.Graph(id = 'grafico-dia-semana'),
        dcc.Graph(id = 'grafico-outliers'),
        dcc.Graph(id = 'grafico-distribuicao'),
        dcc.Graph(id = 'grafico-media-desvio'),
        dcc.Graph(id = 'grafico-acumulado')

    ])
])

# Callback ppara atualizar os graficos conforme os filtros selecionados

@app.callback([
Output('grafico-produto', 'figure'),
Output('grafico-regiao', 'figure'),
Output('grafico-mensal', 'figure'),
Output('grafico-diario', 'figure'),
Output('grafico-dia-semana', 'figure'),
Output('grafico-outliers', 'figure'),
Output('grafico-distribuicao', 'figure'),
Output('grafico-media-desvio', 'figure'),
Output('grafico-acumulado', 'figure')],
[Input('produto-dropdown', 'value'),
Input('regiao-dropdown', 'value'),
Input('ano-dropdown', 'value'),
Input('date-picker-range', 'start_date'),
Input('date-picker-range', 'end_date')])

    
def update_graph(produtos, regioes, ano, start_date, end_date):
    try:
        # converter as datas que chegaram para o formato correto
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # agora atualizamos o graficos com base nos filtros selecionados
        fig_produto = analise.analise_vendas_por_produto(produtos)
        fig_regiao = analise.analise_vendas_por_regiao(regioes)
        fig_mensal = analise.analise_vendas_mensais(ano)
        fig_diario = analise.analise_vendas_diarias(start_date, end_date)
        fig_dia_semana = analise.analise_vendas_por_dia_da_semana()
        fig_distribuicao = analise.distribuicao_vendas()
        fig_outliers = analise.analise_outliers()
        fig_acumulado = analise.vendas_acumuladas()
        media, desvio = analise.analise_media_desvio()
        fig_media_desvio = go.Figure(data=[
            go.Bar(x = ['Média', 'Desvio Padrão'], y = [media, desvio], marker_color = ['blue', 'red'])
        ], layout = go.Layout(title = f'Média e desvio padrão: Média = {media:.2f}, Desvio = {desvio:.2f}'))

        return fig_produto, fig_regiao, fig_mensal, fig_diario, fig_dia_semana, fig_outliers, fig_distribuicao, fig_media_desvio, fig_acumulado

    except Exception as e:
        # caso ocorra algum erro, logar a mensagem de erro e retornar fraficos vazios
        print(f'Erro ao atualizar os graficos: {e}')
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()

# Rodando a aplicação
if __name__ == '__main__':
    app.run(debug=True)