import pandas as pd
import numpy as np
from plotly.offline import init_notebook_mode
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import mannwhitneyu

init_notebook_mode(connected=True)
# %matplotlib inline


def GUS_rel_ch(df, dates):
    sale_id = 'Sale 1' if dates['end'].year == 2017 else 'Sale 2'
    # fill missing ['Date', 'Country'] df with 0 in [BASIS]
    BASIS = 'Gross Units Sold'
    countries_u = df['Country'].unique()
    dates_u = df['Date'].unique()
    fill_missing = pd.MultiIndex.from_product([dates_u, countries_u, [0]], names=['Date', 'Country', BASIS])
    fill_missing = pd.DataFrame().reindex(fill_missing).reset_index()
    df = pd.concat([df, fill_missing])

    # add new column identifying period
    df['Period'] = 'before sale'  # initiate with 'before sale' status
    df.loc[df.Date >= dates['sale_start'], 'Period'] = 'during sale'
    df.loc[df.Date > dates['sale_end'], 'Period'] = 'after sale'

    # sum daily BASIS by period, date and country
    df = df[['Period', 'Date', 'Country', BASIS]].groupby(['Period', 'Date', 'Country']).sum().reset_index()
    # remove countries with abs(mean([Net Units Sold]))<=0.1 in any named periods or less than 50 total [Net Units Sold]
    df['cond1'] = df[['Period', 'Country', BASIS]].groupby(['Period', 'Country']).transform('mean')
    cond1 = df[['Country', 'cond1']].groupby(['Country']).transform(lambda x: np.min(np.abs(x))) > 0.1
    df = df.drop(columns=['cond1'])
    cond2 = df[['Country', BASIS]].groupby(['Country']).transform('sum') > 50
    df = df[np.logical_and(cond1, cond2).values]

    # calculate means per country in periods
    m_before_sale = df.loc[(df.Period == 'before sale'), ['Country', BASIS]].groupby(['Country']).mean()
    m_before_sale = m_before_sale.reset_index().values
    m_before_sale = {x[0]: x[1] for x in m_before_sale}
    # calculate mean during sale by dividing by (number of days - 1),
    # because during the first and last day of the sale, the promotion takes only a fraction of the day
    # for example (19-24) on the first day and (0-19) on the last day
    m_sale = (df.loc[(df.Period == 'during sale'), ['Country', BASIS]].groupby(['Country'])
              .agg(lambda x: np.sum(x) / (len(x) - 1)))
    m_sale = m_sale.reset_index().values
    m_sale = {x[0]: x[1] for x in m_sale}
    # normalize [BASIS] by mean before sale
    df[BASIS + ' by mean'] = df.apply(lambda r: r[BASIS] / m_before_sale[r['Country']], axis=1)

    # calculate normalized mean [BASIS+' by mean'] ratios per country in periods
    nm_before_sale = df.loc[(df.Period == 'before sale'), ['Country', BASIS + ' by mean']].groupby(['Country']).mean()
    nm_before_sale = nm_before_sale.reset_index().values
    nm_before_sale = {x[0]: x[1] for x in nm_before_sale}
    # see description for means above
    nm_sale = (df.loc[(df.Period == 'during sale'), ['Country', BASIS + ' by mean']].groupby(['Country'])
               .agg(lambda x: np.sum(x) / (len(x) - 1)))
    nm_sale = nm_sale.reset_index().values
    nm_sale = {x[0]: x[1] for x in nm_sale}

    ms = 10

    # sort descending on means ratios
    df['norm ' + BASIS + ' means ratio'] = df.apply(
        lambda r: nm_sale[r['Country']] / nm_before_sale[r['Country']], axis=1)
    df = df.sort_values(by=['norm ' + BASIS + ' means ratio'], ascending=False)
    countries = df['Country'].unique()

    # plot GUS data
    fig_in = px.box(df, x='Country', y=BASIS, color_discrete_sequence=['blue', 'green', 'red'],
                    color='Period', category_orders={'Period': ['before sale', 'during sale', 'after sale']})
    fig_in.add_trace(go.Scatter(x=countries, y=[m_before_sale[c] for c in countries], mode='markers',
                                name='mean before sale', showlegend=True, hovertemplate='%{x}: %{y}',
                                marker_symbol='line-ew', marker_line_color='darkblue',
                                marker_color='blue', marker_line_width=2, marker_size=ms))
    fig_in.add_trace(go.Scatter(x=countries, y=[m_sale[c] for c in countries], mode='markers',
                                name='mean during sale', showlegend=True, hovertemplate='%{x}: %{y}',
                                marker_symbol='line-ew', marker_line_color='darkgreen',
                                marker_color='green', marker_line_width=2, marker_size=ms))

    # plot normalized GUS data
    box_traces = px.box(df, x='Country', y=BASIS + ' by mean', color_discrete_sequence=['blue', 'green', 'red'],
                        color='Period', category_orders={'Period': ['before sale', 'during sale', 'after sale']}).data
    box_traces[0]['visible'] = False
    box_traces[1]['visible'] = False
    box_traces[2]['visible'] = False
    fig_in.add_trace(box_traces[0])
    fig_in.add_trace(box_traces[1])
    fig_in.add_trace(box_traces[2])
    fig_in.add_trace(go.Scatter(x=countries, y=[nm_before_sale[c] for c in countries], mode='markers',
                                name='mean before sale', showlegend=True, hovertemplate='%{x}: %{y}',
                                marker_symbol='line-ew', marker_line_color='darkblue',
                                marker_color="blue", marker_line_width=2, marker_size=ms, visible=False))
    fig_in.add_trace(go.Scatter(x=countries, y=[nm_sale[c] for c in countries], mode='markers',
                                name='mean during sale', showlegend=True, hovertemplate='%{x}: %{y}',
                                marker_symbol='line-ew', marker_line_color='darkgreen',
                                marker_color='green', marker_line_width=2, marker_size=ms, visible=False))
    fig_in.add_trace(go.Scatter(x=countries, y=[nm_sale[c] for c in countries], mode='lines',
                                name='', showlegend=False, hovertemplate='', opacity=0.3,
                                line_color='black', line_dash='dot', visible=False))

    # get ratios between before and after sale period normalized means
    nNUSmRatio = df[['Country', 'norm ' + BASIS + ' means ratio']].drop_duplicates().values
    nNUSmRatio = {x[0]: x[1] for x in nNUSmRatio}

    # details: add alternating background and ratio values
    for country in countries:
        i = np.where(np.array([*nNUSmRatio]) == country)[0][0]
        fig_in.add_annotation(x=i, y=0.02 - 0.02 * (i % 2), text=f'<b>{nNUSmRatio[country] - 1:.2g}</b>',
                              showarrow=False, yref='paper')
        colors = ['LightSkyBlue', 'azure']
        fig_in.add_shape(
            type='rect',
            xref='x',
            yref='paper',
            x0=i - 0.5,
            y0=0,
            x1=i + 0.5,
            y1=1,
            opacity=0.2,
            line=dict(width=0),
            fillcolor=colors[0] if i % 2 else colors[1])

    fig_in.add_annotation(x=-0.6, y=0.005, text='<b>means change:</b>', showarrow=False, yref='paper',
                          xanchor='right')

    # initially display the normalized data
    # init_visibility = ([False]*5 + [True]*6)
    init_visibility = ([False] * 5 + ['legendonly'] * 3 + [True] * 3)
    for i, d in enumerate(fig_in.data):
        d['visible'] = init_visibility[i]

    fig_in.update_layout(
        # title_text='Daily <b>'+BASIS+'</b> relative means change (between before- and during-sale periods)<BR>per <b>country</b> (<b>'+sale_id+'</b>)',
        title_text='<b>Performance</b> during <b>' + sale_id + '</b> in each <b>country</b>',
        title_x=0.5,
        title_y=1,
        # width=500,
        height=800,
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99
        ),
        legend_title_text=None,
        yaxis_title_text='average daily ' + BASIS + ' (normalized)',
        updatemenus=[
            dict(
                type='buttons',
                direction='left',
                buttons=list([
                    dict(
                        args=[{'visible': [False] * 5 + ['legendonly'] * 3 + [True] * 3},
                              # args=[{'visible': [False]*5 + [True]*6},
                              {'yaxis': {'title': {'text': 'average daily ' + BASIS + ' (normalized)'}}}],
                        label='divided by mean before sale',
                        method='update'
                    ),
                    dict(
                        args=[{'visible': ['legendonly'] * 3 + [True] * 2 + [False] * 6},
                              # args=[{'visible': [True]*5 + [False]*6},
                              {'yaxis': {'title': {'text': 'average daily ' + BASIS}}}],
                        label='original data',
                        method='update'
                    )
                ]),
                pad={'r': 10, 't': 10},
                showactive=True,
                x=0.0,
                # x=0.07,
                xanchor='left',
                y=1.01,
                yanchor='bottom'
            ),
        ]
    )
    fig_in.show()
    # return nNUSmRatio


def draw_recognition():
    fig = plt.figure(figsize=[12, 6])
    a = fig.add_subplot(1, 3, 1)
    image_before = mpimg.imread('before.png')
    imgplot = plt.imshow(image_before)
    a.set_title('Before the sale')
    plt.axis('off')
    a = fig.add_subplot(1, 3, 2)
    image_after = mpimg.imread('after.png')
    imgplot = plt.imshow(image_after)
    a.set_title('After the sale')
    plt.axis('off')
    a = fig.add_subplot(1, 3, 3)
    image_legend = mpimg.imread('legend.png')
    imgplot = plt.imshow(image_legend)
    plt.axis('off')

    plt.show()


def GUS_stats_diff(df, dates):
    sale_id = 'Sale 1' if dates['end'].year == 2017 else 'Sale 2'
    # fill missing ['Date', 'Country'] df with 0 in [BASIS]
    BASIS = 'Gross Units Sold'
    countries_u = df['Country'].unique()
    dates_u = df['Date'].unique()
    fill_missing = pd.MultiIndex.from_product([dates_u, countries_u, [0]], names=['Date', 'Country', BASIS])
    fill_missing = pd.DataFrame().reindex(fill_missing).reset_index()
    df = pd.concat([df, fill_missing])

    # add new column identifying period
    df['Period'] = 'before sale'  # initiate with 'before sale' status
    df.loc[df.Date >= dates['sale_start'], 'Period'] = 'during sale'
    df.loc[df.Date > dates['sale_end'], 'Period'] = 'after sale'

    # sum daily BASIS by period, date and country
    df = df[['Period', 'Date', 'Country', BASIS]].groupby(['Period', 'Date', 'Country']).sum().reset_index()
    # remove countries with abs(mean([Net Units Sold]))<=0.1 in any named periods or less than 50 total [Net Units Sold]
    df['cond1'] = df[['Period', 'Country', BASIS]].groupby(['Period', 'Country']).transform('mean')
    cond1 = df[['Country', 'cond1']].groupby(['Country']).transform(lambda x: np.min(np.abs(x))) > 0.1
    df = df.drop(columns=['cond1'])
    cond2 = df[['Country', BASIS]].groupby(['Country']).transform('sum') > 50
    df = df[np.logical_and(cond1, cond2).values]

    # estimate probability that before and after sale periods are different
    countries = df['Country'].unique()
    ps = {}
    for country in countries:
        nus_bs = df.loc[(df.Country == country) & (df.Period == 'before sale'), BASIS].values
        nus_as = df.loc[(df.Country == country) & (df.Period == 'after sale'), BASIS].values
        _, ps[country] = mannwhitneyu(nus_bs, nus_as)

    # calculate means per country in periods
    m_before_sale = df.loc[(df.Period == 'before sale'), ['Country', BASIS]].groupby(['Country']).mean()
    m_before_sale = m_before_sale.reset_index().values
    m_before_sale = {x[0]: x[1] for x in m_before_sale}
    m_after_sale = df.loc[(df.Period == 'after sale'), ['Country', BASIS]].groupby(['Country']).mean()
    m_after_sale = m_after_sale.reset_index().values
    m_after_sale = {x[0]: x[1] for x in m_after_sale}
    # normalize [BASIS] by mean before sale
    df[BASIS + ' by mean'] = df.apply(lambda r: r[BASIS] / m_before_sale[r['Country']], axis=1)

    # prepare data to plot
    to_plot = df.loc[(df.Period == 'before sale') | (df.Period == 'after sale')]
    # add column with information if periods before and after sale are statistically different
    to_plot = to_plot.assign(isdiff=to_plot.apply(lambda r: ps[r['Country']] > 0.05, axis=1))
    # calculate normalized mean [BASIS+' by mean'] ratios per country during both periods
    nm_before_sale = to_plot.loc[(to_plot.Period == 'before sale'),
                                 ['Country', BASIS + ' by mean']].groupby(['Country']).mean()
    nm_before_sale = nm_before_sale.reset_index().values
    nm_before_sale = {x[0]: x[1] for x in nm_before_sale}
    nm_after_sale = to_plot.loc[(to_plot.Period == 'after sale'),
                                ['Country', BASIS + ' by mean']].groupby(['Country']).mean()
    nm_after_sale = nm_after_sale.reset_index().values
    nm_after_sale = {x[0]: x[1] for x in nm_after_sale}

    # sort data before plotting: statistically different first, then order by means ratios
    to_plot['norm ' + BASIS + ' means ratio'] = to_plot.apply(
        lambda r: nm_after_sale[r['Country']] / nm_before_sale[r['Country']], axis=1)
    to_plot = to_plot.sort_values(by=['isdiff', 'norm ' + BASIS + ' means ratio'], ascending=(True, False))
    # refresh list of countries, so it has the to_plot ordering
    countries = to_plot['Country'].unique()

    ms = 10
    # plot data
    fig_in = px.box(to_plot, x='Country', y=BASIS, color_discrete_sequence=['blue', 'red'],
                    color='Period', category_orders={'Period': ['before sale', 'after sale']})
    fig_in.add_trace(go.Scatter(x=countries, y=[m_before_sale[c] for c in countries], mode='markers',
                                name='mean before sale', showlegend=True, hovertemplate='%{x}: %{y}',
                                marker_symbol='line-ew', marker_line_color='darkblue',
                                marker_color='blue', marker_line_width=2, marker_size=ms))
    fig_in.add_trace(go.Scatter(x=countries, y=[m_after_sale[c] for c in countries], mode='markers',
                                name='mean after sale', showlegend=True, hovertemplate='%{x}: %{y}',
                                marker_symbol='line-ew', marker_line_color='darkred',
                                marker_color='red', marker_line_width=2, marker_size=ms))

    # plot normalized GUS data
    box_traces = px.box(to_plot, x='Country', y=BASIS + ' by mean', color_discrete_sequence=['blue', 'red'],
                        color='Period', category_orders={'Period': ['before sale', 'after sale']}).data
    box_traces[0]['visible'] = False
    box_traces[1]['visible'] = False
    fig_in.add_trace(box_traces[0])
    fig_in.add_trace(box_traces[1])
    fig_in.add_trace(go.Scatter(x=countries, y=[nm_before_sale[c] for c in countries], mode='markers',
                                name='mean before sale', showlegend=True, hovertemplate='%{x}: %{y}',
                                marker_symbol='line-ew', marker_line_color='darkblue',
                                marker_color="blue", marker_line_width=2, marker_size=ms, visible=False))
    fig_in.add_trace(go.Scatter(x=countries, y=[nm_after_sale[c] for c in countries], mode='markers',
                                name='mean after sale', showlegend=True, hovertemplate='%{x}: %{y}',
                                marker_symbol='line-ew', marker_line_color='darkred',
                                marker_color='red', marker_line_width=2, marker_size=ms, visible=False))

    # get ratios between before and after sale period normalized means
    nNUSmRatio = to_plot[['Country', 'norm ' + BASIS + ' means ratio']].drop_duplicates().values
    nNUSmRatio = {x[0]: x[1] for x in nNUSmRatio}

    for country in countries:
        i = np.where(np.array([*nNUSmRatio]) == country)[0][0]
        colors = ['pink', 'mistyrose']
        if ps[country] < 0.05:
            if '1' in sale_id:
                fig_in.add_annotation(x=i, y=0.02 - 0.02 * (i % 2), text=f'<b>{nNUSmRatio[country] - 1:.1f}</b>',
                                      showarrow=False, yref='paper')
            else:
                fig_in.add_annotation(x=i, y=0.012 - 0.02 * (i % 2), text=f'<b>{nNUSmRatio[country] - 1:.1f}</b>',
                                      showarrow=False, yref='paper')
            colors = ['LightSkyBlue', 'azure']
        fig_in.add_shape(
            type='rect',
            xref='x',
            yref='paper',
            x0=i - 0.5,
            y0=0,
            x1=i + 0.5,
            y1=1,
            opacity=0.2,
            line=dict(width=0),
            fillcolor=colors[0] if i % 2 else colors[1])
    if '1' in sale_id:
        fig_in.add_annotation(x=-0.6, y=0.01, text='<b>means change:</b>', showarrow=False, yref='paper',
                              xanchor='right')
    else:
        fig_in.add_annotation(x=-0.6, y=0.01, text='<b>means change:</b>', showarrow=False, yref='paper',
                              xanchor='right')

    insig_start = 0
    for country in countries:
        if ps[country] >= 0.05:
            break
        insig_start += 1
    insig_end = len(countries) - 1
    fig_in.add_annotation(x=(insig_start + insig_end) / 2., y=0.005,
                          text='NO STATISTICAL DIFFERENCE BETWEEN DISTRIBUTIONS',
                          showarrow=False, yref='paper', xanchor='center')
    fig_in.add_trace(go.Scatter(x=countries[:insig_start], y=[nm_after_sale[c] for c in countries], mode='lines',
                                name='', showlegend=False, hovertemplate='', opacity=0.3,
                                line_color='black', line_dash='dot', visible=False))

    # initially display the normalized data
    init_visibility = ([False] * 4 + ['legendonly'] * 2 + [True] * 3)
    for i, d in enumerate(fig_in.data):
        d['visible'] = init_visibility[i]

    fig_in.update_layout(
        title_text='<b>' + sale_id + '</b> impact on the <b>book\'s recognition</b> per <b>country</b>',
        legend_title_text=None,
        title_x=0.5,
        title_y=1,
        height=800,
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99
        ),
        yaxis_title_text='average daily ' + BASIS + ' (normalized)',
        updatemenus=[
            dict(
                type='buttons',
                direction='left',
                buttons=list([
                    dict(
                        args=[{'visible': [False] * 4 + ['legendonly'] * 2 + [True] * 3},
                              {'yaxis': {'title': {'text': 'average daily ' + BASIS + ' (normalized)'}}}],
                        label='divided by mean before sale',
                        method='update'
                    ),
                    dict(
                        args=[{'visible': ['legendonly'] * 2 + [True] * 2 + [False] * 5},
                              {'yaxis': {'title': {'text': 'average daily ' + BASIS}}}],
                        label='original data',
                        method='update'
                    ),
                ]),
                pad={'r': 10, 't': 10},
                showactive=True,
                x=0.0,
                xanchor='left',
                y=1.01,
                yanchor='bottom'
            ),
        ]
    )
    fig_in.show()


def plot_gauge2(fig_in, means_sale1, means_sale2, what):
    col = 0 if what == 'Gross Units Sold' else 1
    plt.plot([means_sale1['before sale'],
              means_sale1['during sale'],
              means_sale2['before sale'],
              means_sale2['during sale'],
              ])
    plt.gca().set_ylim([0, None])
    tickvals1 = np.sort(np.append(plt.gca().get_yticks(),
                                  [means_sale1['before sale'], means_sale1['during sale']]
                                  ))
    tickvals2 = np.sort(np.append(tickvals1,
                                  [means_sale2['before sale'], means_sale2['during sale']]
                                  ))
    plt.close('all')
    ticknames = {means_sale1['before sale']: '',
                 means_sale1['during sale']: 'during Sale 1',
                 means_sale2['before sale']: '<b>before Sale 2</b><BR>before Sale 1',
                 means_sale2['during sale']: '<b>during Sale 2<BR></b>'
                 }
    ticktext2 = [f'{x:.0f}' if x not in ticknames else ticknames[x] for x in tickvals2]
    gauge_axis = dict(range=[None, tickvals2[-1]],
                      tickmode='array',
                      tickangle=0,
                      tickvals=tickvals2,
                      ticktext=ticktext2
                      )
    gauge_Steps = [{'range': [0, means_sale1['before sale']], 'color': "gray"},
                   {'range': [means_sale1['before sale'], means_sale2['before sale']], 'color': "lightgray"},
                   {'range': [means_sale2['before sale'], means_sale1['during sale']], 'color': "darkseagreen"}]
    fig_in.add_trace(go.Indicator(
        mode='gauge+number+delta',  # draw the gauge, achieved value and it's change
        value=means_sale2['during sale'],
        gauge_steps=gauge_Steps,
        gauge_axis=gauge_axis,
        # Set the reference value to the before-sale mean and delta-mode to relative (shows relative change in %)
        delta={'reference': means_sale2['before sale'], 'relative': True},
        title={'text': 'mean daily <b>' + what + '</b><BR>(<b>Sale 2</b>)'},
        domain={'y': [0.2, 1], 'x': [col * 0.5 + 0.5 * 0.3, col * 0.5 + 0.5 * 0.7]},
    ))
    fig_in.add_trace(go.Indicator(
        mode='delta',
        value=(means_sale2['during sale'] - means_sale2['before sale']) / means_sale2['before sale'],
        delta={'reference': (means_sale1['during sale'] - means_sale1['before sale']) / means_sale1['before sale'],
               'relative': True
               },
        title={'text': '<b>Sale 2</b> vs <b>Sale 1</b>'},
        domain={'y': [0.2, 0.4], 'x': [col * 0.5 + 0.5 * 0.4, col * 0.5 + 0.5 * 0.6]},
    ))


def choropleth2(sales_df, country_codes, pop_data):
    # Take only the during-sale data
    s_during = sales_df.loc[(sales_df.Period == 'during sale')]
    # Sum Gross Units Sold per country code during the sale
    s_during = s_during[['Country Code', 'Gross Units Sold']].groupby(['Country Code']).sum().reset_index()
    s_during = pd.merge(s_during, country_codes, left_on='Country Code', right_on='Alpha2', how='left')
    s_during = s_during.drop(columns=['Alpha2', 'Country Code'])
    s_during = pd.merge(s_during, pop_data, left_on='Alpha3', right_on='Country Code', how='left')
    s_during['Gross Units Sold per 1M pop.'] = s_during.apply(
        lambda r: (r['Gross Units Sold'] / ((r['2017'] + r['2018']) / 2)) * 1e6,
        axis=1
    )
    fig = go.Figure(data=go.Choropleth(
        locations=s_during['Alpha3'],
        z=np.log10(s_during['Gross Units Sold']),
        text=s_during['Country'],
        customdata=s_during['Gross Units Sold'].values,
        hovertemplate='<b>%{text}</b><br>Gross Units Sold: %{customdata}',
        name='',
        colorbar_title='',
        colorbar_tickvals=[0, 1, 2, 3, 4, 5],
        colorbar_ticktext=["1", "10", "100", "1K", '10k', '100k'],
        colorscale='viridis',
        marker_line_color='navy',
        marker_line_width=0.5,
    ))
    map_norm = go.Choropleth(
        locations=s_during['Alpha3'],
        z=np.log10(s_during['Gross Units Sold per 1M pop.']),
        text=s_during['Country'],
        customdata=s_during['Gross Units Sold per 1M pop.'].values,
        hovertemplate='<b>%{text}</b><br>Gross Units Sold per 1M pop.: %{customdata:.1f}',
        name='',
        colorbar_title='',
        colorbar_tickvals=[0, 1, 2, 3],
        colorbar_ticktext=['1', '10', '100', '1K'],
        colorscale='viridis',
        marker_line_color='navy',
        marker_line_width=0.5,
    )
    map_norm['visible'] = False
    fig.add_trace(map_norm)

    fig.update_layout(
        title_text='<b>Gross Units Sold</b> during <b>Sale 2</b>',
        title_x=0.5,
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ),
        updatemenus=[
            dict(
                type='buttons',
                direction='left',
                buttons=list([
                    dict(
                        args=[{'visible': [True] * 1 + [False] * 1}],
                        label='Gross Units Sold',
                        method='update'
                    ),
                    dict(
                        args=[{'visible': [False] * 1 + [True] * 1}],
                        label='Gross Units Sold per 1M pop.',
                        method='update'
                    ),
                ]),
                pad={'r': 10, 't': 10},
                showactive=True,
                x=0.0,
                # x=0.07,
                xanchor='left',
                # y=1.1,
                y=1.01,
                yanchor='bottom'
            ),
        ]
    )
    fig.show()
