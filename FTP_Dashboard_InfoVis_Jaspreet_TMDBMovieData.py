import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import shapiro, kstest, normaltest

np.random.seed(5764)

fontdict = {
    'title': {'font': {'family': 'serif', 'color': 'blue', 'size': 24}},
    'xaxis_title': {'font': {'family': 'serif', 'color': 'darkred', 'size': 20}},
    'yaxis_title': {'font': {'family': 'serif', 'color': 'darkred', 'size': 20}},
    'legend_title': {'font': {'family': 'serif', 'color': 'darkred', 'size': 16}}
}

#
# # -------------Phase II: Interactive web-based dashboard-------------
#

# Data Cleaning

loc = 'data/TMDB_movie_dataset_v11.csv'
data = pd.read_csv(loc)

# Convert release_date to datetime format
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')

# Drop rows with NaN release_date
data_cleaned = data.dropna(subset=['release_date'])
data = data.drop(data[(data['adult'] == True)].index)

# Sort the data by release_date
data_cleaned = data_cleaned.sort_values(by='release_date')

# Convert numerical features to numeric, coercing errors
numerical_features = ['vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity']
for feature in numerical_features:
    data_cleaned[feature] = pd.to_numeric(data_cleaned[feature], errors='coerce')

data_cleaned = data_cleaned.dropna(subset=numerical_features)

data_cleaned['original_language'] = data_cleaned['original_language'].fillna('Unknown')

# Adding Release Year column
data_cleaned['release_year'] = data_cleaned['release_date'].dt.year

data_cleaned = data_cleaned[data_cleaned['vote_average'] > 0]

data_exploded = data_cleaned.assign(genres=data_cleaned['genres'].str.split(', ')).explode('genres')

# Get top 8 genres
top_genres = data_exploded['genres'].value_counts().head(8).index.tolist()
data_top_genres = data_exploded[data_exploded['genres'].isin(top_genres)]

# Split production companies into individual entries
data_exploded_companies = data_cleaned.assign(
    production_companies=data_cleaned['production_companies'].str.split(', ')).explode('production_companies')

# Get top 8 production companies
top_companies = data_exploded_companies['production_companies'].value_counts().nlargest(8).index.tolist()
data_top_companies = data_exploded_companies[data_exploded_companies['production_companies'].isin(top_companies)]


# -------------------------------------------------------

# Dashboard

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash('Info Vis Project', external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# server = app.server

# Function for normality tests
def shapiro_test(x, title):
    stats, p = shapiro(x)
    alpha = 0.01
    result = f'Shapiro test result: statistics = {stats:.2f}, p-value = {p:.2f}.\n'
    if p > alpha:
        result += f'\n{title} feature is normally distributed (fail to reject H0)'
    else:
        result += f'\n{title} feature is NOT normally distributed (reject H0)'
    return result


def ks_test(x, title):
    mean = np.mean(x)
    std = np.std(x)
    dist = np.random.normal(mean, std, len(x))
    stats, p = kstest(x, dist)
    alpha = 0.01
    result = f'K-S test result: statistics = {stats:.2f}, p-value = {p:.2f}.\n'
    if p > alpha:
        result += f'\n{title} feature is normally distributed (fail to reject H0)'
    else:
        result += f'\n{title} feature is NOT normally distributed (reject H0)'
    return result


def da_k_squared_test(x, title):
    stats, p = normaltest(x)
    alpha = 0.01
    result = f'Da K-squared test result: statistics = {stats:.2f}, p-value = {p:.2f}.\n'
    if p > alpha:
        result += f'\n{title} feature is normally distributed (fail to reject H0)'
    else:
        result += f'\n{title} feature is NOT normally distributed (reject H0)'
    return result


app.layout = html.Div([
    html.H1('TMDB Movies Dashboard', style={'textAlign': 'center'}),
    html.Br(),
    dcc.Tabs(id='tabs', children=[
        dcc.Tab(label='Data Analysis', value='tab-data-analysis'),
        dcc.Tab(label='Correlation Analysis', value='tab-correlation-analysis'),
        dcc.Tab(label='Genre Distribution', value='tab-genre-distribution'),
        dcc.Tab(label='Release Year', value='tab-release-year'),
        dcc.Tab(label='Distribution Plots', value='tab-distribution'),
        dcc.Tab(label='Scatter Plots', value='tab-scatterplot'),
        dcc.Tab(label='Temporal Analysis', value='tab-temporal'),
        dcc.Tab(label='Vote Analysis', value='tab-vote-analysis'),
        dcc.Tab(label='Feedback', value='tab-feedback')
    ]),
    html.Div(id='tab-content')
])


# -------------------------------------------------------


# Data Analysis tab
data_analysis_layout = html.Div([
    html.H1('Data Analysis'),
    html.H2('Outlier Detection and Removal'),
    html.Label('Choose the features for the box plot'),
    dcc.Checklist(
        id='outlier-feature-checklist',
        options=[{'label': feature, 'value': feature} for feature in numerical_features],
        value=['vote_average', 'vote_count']
    ),
    html.Br(),
    html.Label('Select to remove outliers'),
    dcc.Checklist(
        id='remove-outliers',
        options=[{'label': 'Remove Outliers', 'value': 'remove'}],
        value=['remove']
    ),
    dcc.Graph(id='outlier-boxplot'),
    html.H2('Normality Test'),
    html.Label('Select the feature'),
    dcc.Dropdown(
        id='normality-feature-dropdown',
        options=[{'label': feature, 'value': feature} for feature in numerical_features],
        value='revenue'
    ),
    html.Br(),
    dcc.RadioItems(
        id='normality-test-type',
        options=[
            {'label': 'Shapiro Test', 'value': 'shapiro'},
            {'label': 'Kolmogorov-Smirnov Test', 'value': 'ks'},
            {'label': 'D’Agostino-Pearson Test', 'value': 'da'}
        ],
        value='shapiro'
    ),
    html.Br(),
    html.H5(id='normality-test-result'),
    html.Button(id='run-normality-test', n_clicks=0, children='Run Normality Test')
])


@app.callback(
    Output('outlier-boxplot', 'figure'),
    [Input('outlier-feature-checklist', 'value'), Input('remove-outliers', 'value')]
)
def update_outlier_boxplot(selected_features, remove_outliers):
    filtered_df = data_cleaned[data_cleaned[selected_features] > 0]
    if 'remove' in remove_outliers:
        for feature in selected_features:
            Q1 = filtered_df[feature].quantile(0.25)
            Q3 = filtered_df[feature].quantile(0.75)
            IQR = Q3 - Q1
            filtered_df = filtered_df[
                ~((filtered_df[feature] < (Q1 - 1.5 * IQR)) | (filtered_df[feature] > (Q3 + 1.5 * IQR)))]
    fig = px.box(filtered_df, y=selected_features, title='Boxplot for Outlier Detection and Removal')
    fig.update_layout(
        title=fontdict['title'],
        xaxis_title={'text': 'Features', **fontdict['xaxis_title']},
        yaxis_title={'text': 'Values', **fontdict['yaxis_title']},
        legend_title=fontdict['legend_title'],
        template='plotly_white'
    )
    fig.update_traces(marker=dict(color='lightblue', line=dict(width=2)))
    return fig


@app.callback(
    Output('normality-test-result', 'children'),
    [Input('run-normality-test', 'n_clicks')],
    [State('normality-feature-dropdown', 'value'), State('normality-test-type', 'value')]
)
def run_normality_test(n_clicks, selected_feature, test_type):
    if n_clicks > 0:
        filtered_df = data_cleaned[data_cleaned[selected_feature] > 0]
        if test_type == 'shapiro':
            result = shapiro_test(filtered_df[selected_feature], selected_feature)
        elif test_type == 'ks':
            result = ks_test(filtered_df[selected_feature], selected_feature)
        elif test_type == 'da':
            result = da_k_squared_test(filtered_df[selected_feature], selected_feature)
        return result
    return ""


# -------------------------------------------------------


# Correlation Analysis tab
correlation_analysis_layout = html.Div([
    html.H1('Correlation Analysis'),
    html.H6('Select Features for Analysis'),
    dcc.Dropdown(
        id='correlation-feature-dropdown',
        options=[{'label': feature, 'value': feature} for feature in numerical_features],
        value=numerical_features,
        multi=True
    ),
    html.H2('Pearson Correlation Heatmap'),
    dcc.Graph(id='correlation-heatmap'),
    html.H2('Scatter Plot Matrix'),
    dcc.Graph(id='scatter-matrix')
])


@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('correlation-feature-dropdown', 'value')]
)
def update_correlation_heatmap(selected_features):
    corr = data_cleaned[selected_features].corr()
    fig = px.imshow(corr, text_auto='.2f', title='Pearson Correlation Coefficient Heatmap')
    fig.update_layout(
        title=fontdict['title'],
        template='plotly_white'
    )
    return fig


@app.callback(
    Output('scatter-matrix', 'figure'),
    [Input('correlation-feature-dropdown', 'value')]
)
def update_scatter_matrix(selected_features):
    fig = px.scatter_matrix(data_cleaned[selected_features], title='Scatter Plot Matrix')
    fig.update_layout(
        title=fontdict['title'],
        template='plotly_white'
    )
    return fig


# -------------------------------------------------------


# Genre Distribution tab
genre_distribution_layout = html.Div([
    html.H1('Genre Distribution'),
    html.H6('Select the Genre'),
    dcc.Dropdown(
        id='genre-dropdown',
        options=[{'label': genre, 'value': genre} for genre in top_genres],
        value=top_genres[0]
    ),
    dcc.Loading(
        id='loading-genre-graph',
        type='default',
        children=dcc.Graph(id='genre-graph')
    ),
    html.Button("Download Data", id="btn-download"),
    dcc.Download(id="download-dataframe-csv")
])


@app.callback(
    Output('genre-graph', 'figure'),
    [Input('genre-dropdown', 'value')]
)
def update_genre_graph(selected_genre):
    filtered_df = data_top_genres[data_top_genres['genres'] == selected_genre]
    fig = px.histogram(filtered_df, x='vote_average', nbins=20, title=f'Vote Distribution for {selected_genre} Movies')
    fig.update_layout(
        title=fontdict['title'],
        xaxis_title={'text': 'Vote Average', **fontdict['xaxis_title']},
        yaxis_title={'text': 'Count', **fontdict['yaxis_title']},
        legend_title=fontdict['legend_title'],
        template='plotly_white'
    )
    # Add custom hover information (Tooltips)
    fig.update_traces(
        hovertemplate='<b>Vote Average: %{x}</b><br>Count: %{y}<br>Genre: ' + selected_genre + '<extra></extra>',
        marker = dict(color='lightblue', line=dict(width=2))
    )
    return fig


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-download", "n_clicks"),
    State('genre-dropdown', 'value'),
    prevent_initial_call=True
)
def func(n_clicks, selected_genre):
    filtered_df = data_top_genres[data_top_genres['genres'] == selected_genre]
    return dcc.send_data_frame(filtered_df.to_csv, f"{selected_genre}_movies.csv")


# -------------------------------------------------------


# Release Year tab
release_year_layout = html.Div([
    html.H1('Release Year'),
    html.Label('Select the year range'),
    dcc.RangeSlider(
        id='year-slider',
        min=1900,
        max=2024,
        step=1,
        value=[2000, 2024],
        marks={year: str(year) for year in
               range(1900, 2025, 5)}

    ),
    html.Br(),
    html.Label('Select the genres'),
    dcc.Dropdown(
        id='release-year-genre-dropdown',
        options=[{'label': genre, 'value': genre} for genre in top_genres],
        value=["Drama", "Documentary", "Comedy"],
        multi=True
    ),
    html.Br(),
    html.Label('Choose the plot type'),
    dcc.RadioItems(
        id='bar-plot-type',
        options=[
            {'label': 'Grouped', 'value': 'group'},
            {'label': 'Stacked', 'value': 'stack'}
        ],
        value='group',
        inline=True
    ),
    dcc.Graph(id='year-graph')
])


@app.callback(
    Output('year-graph', 'figure'),
    [Input('year-slider', 'value'),
     Input('release-year-genre-dropdown', 'value'),
     Input('bar-plot-type', 'value')]
)
def update_year_graph(year_range, selected_genres, plot_type):
    start_year, end_year = year_range
    filtered_df = data_top_genres[
        (data_top_genres['release_year'] >= start_year) &
        (data_top_genres['release_year'] <= end_year) &
        (data_top_genres['genres'].isin(selected_genres))
        ]
    fig = px.histogram(filtered_df, x='release_year', color='genres', barmode=plot_type,
                       title=f'Movies Released between {start_year} and {end_year}')
    fig.update_layout(
        title=fontdict['title'],
        xaxis_title={'text': 'Release Year', **fontdict['xaxis_title']},
        yaxis_title={'text': 'Count', **fontdict['yaxis_title']},
        legend_title=fontdict['legend_title'],
        template='plotly_white'
    )
    fig.update_traces(marker=dict(line=dict(width=2)))
    return fig


# -------------------------------------------------------


# Distribution Plots tab
distribution_layout = html.Div([
    html.H1('Distribution Plots'),
    html.Label('Select the feature'),
    dcc.Dropdown(
        id='distribution-feature-dropdown',
        options=[{'label': feature, 'value': feature} for feature in numerical_features],
        value='revenue'
    ),
    dcc.Graph(id='distribution-graph')
])


@app.callback(
    Output('distribution-graph', 'figure'),
    [Input('distribution-feature-dropdown', 'value')]
)
def update_distribution_graph(selected_feature):
    filtered_df = data_cleaned[data_cleaned[selected_feature] > 0]
    filtered_df[f'log_{selected_feature}'] = np.log10(filtered_df[selected_feature])
    fig = px.histogram(filtered_df, x=f'log_{selected_feature}', nbins=50,
                       title=f'Log Distribution of {selected_feature.capitalize()} (Non-Zero Values)')

    # Update x-axis to show the actual values
    tickvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ticktext = [f'{10 ** i:.0f}' for i in tickvals]

    fig.update_layout(
        title=fontdict['title'],
        xaxis=dict(
            tickvals=tickvals,
            ticktext=ticktext,
            title={'text': f'{selected_feature.capitalize()} (Actual Values)', **fontdict['xaxis_title']}
        ),
        yaxis_title=fontdict['yaxis_title'],
        legend_title=fontdict['legend_title'],
        template='plotly_white'
    )
    fig.update_traces(marker=dict(color='lightblue', line=dict(width=2)))
    return fig


# -------------------------------------------------------


# Scatter Plots tab
scatterplot_layout = html.Div([
    html.H1('Scatter Plots'),
    html.Label('Select the feature-X'),
    dcc.Dropdown(
        id='correlation-feature-x',
        options=[{'label': feature, 'value': feature} for feature in numerical_features],
        value='budget'
    ),
    html.Label('Select the feature-Y'),
    dcc.Dropdown(
        id='correlation-feature-y',
        options=[{'label': feature, 'value': feature} for feature in numerical_features],
        value='revenue'
    ),
    html.Label('Select the Genres'),
    dcc.Dropdown(
        id='correlation-genre-dropdown',
        options=[{'label': genre, 'value': genre} for genre in top_genres],
        value=top_genres,
        multi=True
    ),
    html.Label('Select the Companies'),
    dcc.Dropdown(
        id='correlation-company-dropdown',
        options=[{'label': company, 'value': company} for company in top_companies],
        value=top_companies,
        multi=True
    ),
    html.Label('Select the time range'),
    dcc.RangeSlider(
        id='correlation-year-slider',
        min=1900,
        max=2024,
        step=1,
        value=[2000, 2024],
        marks={year: str(year) for year in
               range(1900, 2025, 5)}

    ),
    dcc.Graph(id='correlation-graph')
])


@app.callback(
    Output('correlation-graph', 'figure'),
    [
        Input('correlation-feature-x', 'value'),
        Input('correlation-feature-y', 'value'),
        Input('correlation-genre-dropdown', 'value'),
        Input('correlation-company-dropdown', 'value'),
        Input('correlation-year-slider', 'value')
    ]
)
def update_correlation_graph(feature_x, feature_y, selected_genres, selected_companies, year_range):
    start_year, end_year = year_range
    filtered_df = data_top_companies[
        (data_top_companies['release_year'] >= start_year) &
        (data_top_companies['release_year'] <= end_year) &
        (data_top_companies['genres'].isin(selected_genres)) &
        (data_top_companies['production_companies'].isin(selected_companies))
        ]
    fig = px.scatter(filtered_df, x=feature_x, y=feature_y, color='genres',
                     title=f'{feature_x.capitalize()} vs. {feature_y.capitalize()}')
    fig.update_layout(
        title=fontdict['title'],
        xaxis_title={'text': feature_x.capitalize(), **fontdict['xaxis_title']},
        yaxis_title={'text': feature_y.capitalize(), **fontdict['yaxis_title']},
        legend_title=fontdict['legend_title'],
        template='plotly_white'
    )
    fig.update_traces(marker=dict(line=dict(width=2)))
    return fig


# -------------------------------------------------------


# Temporal Analysis tab
temporal_layout = html.Div([
    html.H1('Temporal Analysis'),
    html.Label('Select the time range'),
    dcc.RangeSlider(
        id='temporal-year-slider',
        min=1900,
        max=2024,
        step=1,
        value=[2000, 2024],
        marks={year: str(year) for year in
               range(1900, 2025, 5)}

    ),
    html.Label('Select the feature'),
    dcc.Dropdown(
        id='temporal-feature-dropdown',
        options=[{'label': feature, 'value': feature} for feature in numerical_features],
        value='revenue'
    ),
    html.Label('Select the genres'),
    dcc.Dropdown(
        id='temporal-genre-dropdown',
        options=[{'label': genre, 'value': genre} for genre in top_genres],
        value=top_genres,
        multi=True
    ),
    dcc.Graph(id='temporal-graph')
])


@app.callback(
    Output('temporal-graph', 'figure'),
    [
        Input('temporal-year-slider', 'value'),
        Input('temporal-feature-dropdown', 'value'),
        Input('temporal-genre-dropdown', 'value')
    ]
)
def update_temporal_graph(year_range, selected_feature, selected_genres):
    start_year, end_year = year_range
    filtered_df = data_top_genres[
        (data_top_genres['release_year'] >= start_year) &
        (data_top_genres['release_year'] <= end_year) &
        (data_top_genres['genres'].isin(selected_genres))
        ]
    # mean_values = filtered_df.groupby('release_year')[selected_feature].mean().reset_index()
    # fig = px.line(mean_values, x='release_year', y=selected_feature, title=f'{selected_feature.capitalize()} Over Time')
    mean_values = filtered_df.groupby(['release_year', 'genres'])[selected_feature].mean().reset_index()
    fig = px.line(mean_values, x='release_year', y=selected_feature, color='genres',
                  title=f'{selected_feature.capitalize()} Over Time')
    fig.update_layout(
        title=fontdict['title'],
        xaxis_title={'text': 'Release Year', **fontdict['xaxis_title']},
        yaxis_title={'text': selected_feature.capitalize(), **fontdict['yaxis_title']},
        legend_title=fontdict['legend_title'],
        template='plotly_white'
    )
    fig.update_traces(line=dict(width=2))
    return fig


# -------------------------------------------------------


# Vote Analysis tab
vote_analysis_layout = html.Div([
    html.H1('Vote Analysis'),
    html.Label('Select the minimum vote average'),
    dcc.Slider(id='vote-slider', min=0, max=10, step=0.5, value=5),
    dcc.Graph(id='vote-graph')
])


@app.callback(
    Output('vote-graph', 'figure'),
    [Input('vote-slider', 'value')]
)
def update_vote_graph(slider_value):
    filtered_df = data_cleaned[data_cleaned['vote_average'] >= slider_value]
    fig = px.histogram(filtered_df, x='vote_average', nbins=20, title=f'Movies with Vote Average >= {slider_value}')
    fig.update_layout(
        title=fontdict['title'],
        xaxis_title={'text': 'Vote Average', **fontdict['xaxis_title']},
        yaxis_title={'text': 'Count', **fontdict['yaxis_title']},
        legend_title=fontdict['legend_title'],
        template='plotly_white'
    )
    fig.update_traces(marker=dict(line=dict(width=2)))
    return fig


# -------------------------------------------------------


# Feedback tab
feedback_layout = html.Div([
    html.H1('Feedback'),
    html.Img(src='/assets/FeedbackImage.jpg', alt='Feedback Image',
             style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
    html.H3('We value your feedback!', style={'textAlign': 'center'}),
    dcc.Textarea(
        id='feedback-textarea',
        placeholder='Enter your feedback here...',
        style={'width': '100%', 'height': 200}
    ),
    html.Div(id='feedback-output', style={'textAlign': 'center', 'margin-top': 20}),
    html.Button('Submit', id='submit-feedback', n_clicks=0,
                style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'})
])


@app.callback(
    Output('feedback-output', 'children'),
    [Input('submit-feedback', 'n_clicks')],
    [State('feedback-textarea', 'value')]
)
def update_feedback(n_clicks, feedback_value):
    if n_clicks > 0:
        return f'Thank you for your feedback!!'
    return ""


# -------------------------------------------------------


@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value')]
)
def render_content(tab):
    if tab == 'tab-data-analysis':
        return data_analysis_layout
    elif tab == 'tab-correlation-analysis':
        return correlation_analysis_layout
    elif tab == 'tab-genre-distribution':
        return genre_distribution_layout
    elif tab == 'tab-release-year':
        return release_year_layout
    elif tab == 'tab-distribution':
        return distribution_layout
    elif tab == 'tab-scatterplot':
        return scatterplot_layout
    elif tab == 'tab-temporal':
        return temporal_layout
    elif tab == 'tab-vote-analysis':
        return vote_analysis_layout
    elif tab == 'tab-feedback':
        return feedback_layout


if __name__ == '__main__':
    app.run_server(debug=True)
